import numpy as np


JOINTS_IND = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 34, 40, 49]

def recover_motion(motion_data, initial_root=None, initial_yaw=0.0):
    """
    Reconstruct absolute joint positions from the (T, 85) feature array.

    Parameters
    ----------
    motion_data : np.ndarray, shape (T, 85)
        Each row contains one delta-yaw scalar followed by 28*3 local joint coords.
    initial_root : array-like, shape (3,), optional
        Root position at frame 0. Defaults to the origin.
    initial_yaw : float, optional
        Root yaw (radians) at frame 0. Defaults to 0.

    Returns
    -------
    np.ndarray, shape (T, 28, 3)
        Reconstructed absolute joint positions.
    """
    point_cloud = motion_data[:, 1:].reshape(-1, 28, 3)
    delta_yaw = motion_data[:, 0]

    if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
        raise ValueError("`feature` must have shape (T, J, 3).")

    T, J, _ = point_cloud.shape
    reconstructed = np.empty_like(point_cloud)

    if initial_root is None:
        prev_root_ground = np.zeros(3, dtype=np.float64)
    else:
        prev_root_ground = np.asarray(initial_root, dtype=np.float64).copy()
    prev_root_ground[1] = 0.0
    prev_yaw = float(initial_yaw)

    for t in range(T):
        frame_feat = point_cloud[t]

        x_local = frame_feat[:, 0]
        z_local = frame_feat[:, 2]
        y_abs = frame_feat[:, 1]

        cos_yaw = np.cos(prev_yaw)
        sin_yaw = np.sin(prev_yaw)

        rel_x = cos_yaw * x_local + sin_yaw * z_local
        rel_z = -sin_yaw * x_local + cos_yaw * z_local

        frame_global = np.empty_like(frame_feat)
        frame_global[:, 0] = rel_x + prev_root_ground[0]
        frame_global[:, 1] = y_abs
        frame_global[:, 2] = rel_z + prev_root_ground[2]

        reconstructed[t] = frame_global

        current_root = frame_global[0]

        curr_yaw = prev_yaw + delta_yaw[t]

        prev_yaw = curr_yaw
        prev_root_ground = current_root.copy()
        prev_root_ground[1] = 0.0

    return reconstructed