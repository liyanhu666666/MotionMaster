import torch
import torch.nn.functional as F

from typing import NewType

Tensor = NewType('Tensor', torch.Tensor)

REST_JOINTS = torch.tensor([[ 1.1677e-03, -3.6684e-01,  1.2669e-02],
        [ 5.7311e-02, -4.6138e-01, -1.0805e-02],
        [-5.6702e-02, -4.7201e-01, -3.8898e-03],
        [-1.6842e-04, -2.5642e-01, -2.5256e-02],
        [ 1.2454e-01, -8.5825e-01, -1.7460e-02],
        [-1.0743e-01, -8.5183e-01, -1.8341e-02],
        [ 1.0041e-02, -1.0545e-01, -2.0813e-02],
        [ 7.8947e-02, -1.2795e+00, -5.8606e-02],
        [-9.0274e-02, -1.2867e+00, -5.8279e-02],
        [ 1.0488e-03, -4.7606e-02,  1.8564e-03],
        [ 1.2326e-01, -1.3403e+00,  7.6550e-02],
        [-1.2572e-01, -1.3464e+00,  8.3534e-02],
        [-8.5336e-03,  1.1842e-01, -2.5384e-02],
        [ 4.3469e-02,  2.8671e-02, -3.3985e-03],
        [-4.5028e-02,  2.9210e-02, -6.6726e-03],
        [ 1.4370e-02,  2.7913e-01, -2.4442e-03],
        [ 1.8442e-01,  8.9088e-02, -1.8298e-02],
        [-1.7538e-01,  8.8060e-02, -1.9259e-02],
        [ 4.3916e-01,  1.2923e-02, -6.3790e-02],
        [-4.4788e-01,  4.2134e-02, -5.1688e-02],
        [ 7.1077e-01,  3.8536e-02, -6.4438e-02],
        [-7.1126e-01,  4.2163e-02, -6.3510e-02],
        [-6.3344e-03,  2.7825e-01, -8.6245e-03],
        [ 3.2475e-02,  3.2455e-01,  6.3394e-02],
        [-3.2476e-02,  3.2455e-01,  6.3393e-02],
        [ 8.1685e-01,  2.9950e-02, -4.3381e-02],
        [ 8.5019e-01,  3.1590e-02, -3.9728e-02],
        [ 8.7369e-01,  2.8885e-02, -3.9950e-02],
        [ 8.2499e-01,  3.2684e-02, -6.8399e-02],
        [ 8.5684e-01,  3.3105e-02, -7.2849e-02],
        [ 8.8119e-01,  3.0575e-02, -7.7281e-02],
        [ 7.9891e-01,  2.3788e-02, -1.1093e-01],
        [ 8.1590e-01,  2.2050e-02, -1.2297e-01],
        [ 8.3256e-01,  2.0032e-02, -1.3417e-01],
        [ 8.1269e-01,  2.9587e-02, -9.3380e-02],
        [ 8.4227e-01,  3.0226e-02, -9.8451e-02],
        [ 8.6621e-01,  2.7890e-02, -1.0529e-01],
        [ 7.5249e-01,  1.8459e-02, -3.6734e-02],
        [ 7.6963e-01,  2.0286e-02, -9.4506e-03],
        [ 7.9056e-01,  1.5541e-02,  7.5935e-03],
        [-8.1685e-01,  2.9952e-02, -4.3379e-02],
        [-8.5019e-01,  3.1592e-02, -3.9725e-02],
        [-8.7368e-01,  2.8886e-02, -3.9948e-02],
        [-8.2499e-01,  3.2686e-02, -6.8399e-02],
        [-8.5684e-01,  3.3105e-02, -7.2850e-02],
        [-8.8119e-01,  3.0575e-02, -7.7282e-02],
        [-7.9891e-01,  2.3788e-02, -1.1093e-01],
        [-8.1591e-01,  2.2050e-02, -1.2297e-01],
        [-8.3256e-01,  2.0031e-02, -1.3417e-01],
        [-8.1269e-01,  2.9589e-02, -9.3381e-02],
        [-8.4227e-01,  3.0228e-02, -9.8450e-02],
        [-8.6621e-01,  2.7891e-02, -1.0529e-01],
        [-7.5249e-01,  1.8457e-02, -3.6732e-02],
        [-7.6962e-01,  2.0285e-02, -9.4495e-03],
        [-7.9055e-01,  1.5542e-02,  7.5940e-03]]).to('cuda')

PARENTS = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
        53]).to('cuda')


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def batch_rigid_transform(
    body_pose: Tensor,
    global_orient: Tensor,
    transl: Tensor,
    dtype=torch.float32,
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    pose : torch.tensor BxNx3
        Tensor of rotvec
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    """
    orig_shape = body_pose.shape[:-1]
    if len(body_pose.shape) > 2:
        body_pose = body_pose.view(-1, 63)
        transl = transl.view(-1, 3)
        global_orient = global_orient.view(-1, 3)

    batch_size = body_pose.shape[0]
    device = body_pose.device
    pose = torch.cat([global_orient.unsqueeze(1), body_pose.reshape(-1, 21, 3), torch.zeros(batch_size, 33, 3).to(device)], dim=1)

    joints = REST_JOINTS.to(device)
    parents = PARENTS.to(device)
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
        [batch_size, -1, 3, 3])

    joints = joints.unsqueeze(0).repeat(batch_size, 1, 1)
    joints = joints.unsqueeze(-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    posed_joints = transforms[:, :, :3, 3]
    posed_joints = posed_joints + transl.unsqueeze(1)

    posed_joints = posed_joints.reshape(*orig_shape, -1, 3)

    return posed_joints