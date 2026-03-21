import torch
import os
from pytorch3d.transforms import matrix_to_axis_angle
import numpy as np
from pytorch3d.transforms import so3_exp_map, so3_log_map
from smplx2joints import batch_rigid_transform

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

_ROOT = os.path.join(os.path.dirname(__file__), '..')

PELVIS_SHIFT = np.array([0.001144, -0.366919, 0.012666])
ROOT_JOINT = np.array([[2.371647860854864e-05, 7.81714916229248e-05, 3.073364496231079e-06],
                         [0.05616740882396698, -0.0944635272026062, -0.023471474647521973],
                         [-0.057845935225486755, -0.10508871078491211, -0.01655576191842556]])
JOINTS_IND = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 34, 40, 49]

def angular_velocity_loss(
    rot_vec: torch.Tensor,
    dt: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Penalize angular velocity using relative rotations (PyTorch3D).

    Args:
        rot_vec (torch.Tensor): (T, J, 3) axis-angle vectors.
        dt (float): Time step between frames.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: Scalar loss (or tensor if reduction='none').
    """
    if rot_vec.ndim != 3 or rot_vec.size(-1) != 3:
        raise ValueError("rot_vec must have shape (T, J, 3).")

    T, J, _ = rot_vec.shape
    if T < 2:
        raise ValueError("Need at least two frames to compute angular velocity.")

    R = so3_exp_map(rot_vec.reshape(-1, 3)).reshape(T, J, 3, 3)
    R_rel = torch.matmul(R[1:], R[:-1].transpose(-1, -2))
    rel_rot_vec = so3_log_map(R_rel.reshape(-1, 3, 3)).reshape(T - 1, J, 3)
    angular_velocity = rel_rot_vec / dt
    velocity_mag_sq = angular_velocity.pow(2).sum(dim=-1)

    if reduction == "mean":
        return velocity_mag_sq.mean()
    if reduction == "sum":
        return velocity_mag_sq.sum()
    if reduction == "none":
        return velocity_mag_sq


def batch_kabsch(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Kabsch alignment.
    src: (3, 3) template joints
    dst: (L, 3, 3) target joints
    """
    L = dst.shape[0]
    src_exp = src.unsqueeze(0).repeat(L, 1, 1)
    src_centered = src_exp - src_exp.mean(dim=1, keepdim=True)
    dst_centered = dst - dst.mean(dim=1, keepdim=True)

    cov = src_centered.transpose(1, 2) @ dst_centered
    U, _, Vh = torch.linalg.svd(cov)
    V = Vh.transpose(-2, -1)
    R = V @ U.transpose(-2, -1)
    det = torch.det(R)
    mask = det < 0
    if mask.any():
        V[mask, :, -1] *= -1
        R = V @ U.transpose(-2, -1)

    src_mean = src_exp.mean(dim=1)
    dst_mean = dst.mean(dim=1)
    T = dst_mean - (R @ src_mean.unsqueeze(-1)).squeeze(-1)
    return R, T

def fit_smplx_from_joints(
    joints_3d: torch.Tensor,
    num_iters: int = 200,
    lr: float = 5e-2,
    model_path=None,
    device: str = None,
) -> dict[str, torch.Tensor]:
    """
    joints_3d: (L, J, 3) target joints for one motion clip.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    joints_3d = joints_3d.to(device)
    L, _, _ = joints_3d.shape

    if model_path is None:
        model_path = os.path.join(_ROOT, 'checkpoints', 'smplx_model')
    vp_model_path = os.path.join(_ROOT, 'src', 'human_body_prior_repo', 'support_data', 'dowloads', 'V02_05')
    
    vposer, _ = load_model(vp_model_path, model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vposer = vposer.to(device)
    vposer.eval()

    template_three = torch.from_numpy(ROOT_JOINT).float().to(device)
    pelvis_shift = torch.from_numpy(PELVIS_SHIFT).float().to(device)

    init_rot_mats, init_trans = batch_kabsch(template_three, joints_3d[:, :3, :])
    init_trans = init_trans - pelvis_shift
    
    global_orient = torch.nn.Parameter(matrix_to_axis_angle(init_rot_mats))
    transl = torch.nn.Parameter(init_trans)
    pose_embedding = torch.nn.Parameter(torch.zeros(L, 32).to(device))

    optimizer = torch.optim.Adam([global_orient, transl, pose_embedding], lr=lr)
    w_prior = 0.04

    for i in range(num_iters // 2):
        optimizer.zero_grad()

        decoded_pose = vposer.decode(pose_embedding)
        body_pose = decoded_pose['pose_body'].reshape(L, 63)

        pred_joints = batch_rigid_transform(body_pose, global_orient, transl)[:, JOINTS_IND, :]

        loss = ((pred_joints - joints_3d) ** 2).sum(dim=-1).mean()

        body_pose_j = torch.cat([body_pose.reshape(L, -1, 3), global_orient.reshape(L, -1, 3)], dim=1)
        loss += angular_velocity_loss(body_pose_j) * 0.35
        loss += torch.mean(pose_embedding ** 2) * w_prior

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        decoded_pose = vposer.decode(pose_embedding)
        body_pose_tensor = decoded_pose['pose_body'].reshape(L, 63).detach().clone()

    body_pose = torch.nn.Parameter(body_pose_tensor, requires_grad=True)

    optimizer_new = torch.optim.Adam([global_orient, transl, body_pose], lr=5e-3)
    
    for i in range(200):
        optimizer_new.zero_grad()
        pred_joints = batch_rigid_transform(body_pose, global_orient, transl)[:, JOINTS_IND, :]
        loss = ((pred_joints - joints_3d) ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer_new.step()

    return {
        "global_orient": global_orient.detach().cpu().numpy(),
        "transl": transl.detach().cpu().numpy(),
        "body_pose": body_pose.detach().cpu().numpy(),
    }

