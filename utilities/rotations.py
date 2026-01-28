import numpy as np
import torch


def quat_wxyz_from_rotmat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (N,3,3) to quaternion (N,4) in wxyz."""
    # Robust conversion, torch-only
    # Based on standard numerically stable branches
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros((R.shape[0], 4), device=R.device, dtype=R.dtype)

    # trace > 0
    mask = tr > 0.0
    if mask.any():
        S = torch.sqrt(tr[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * S
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / S
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / S
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / S

    # otherwise, find max diagonal
    mask2 = ~mask
    if mask2.any():
        R2 = R[mask2]
        diag = torch.stack([R2[:, 0, 0], R2[:, 1, 1], R2[:, 2, 2]], dim=1)
        idx = torch.argmax(diag, dim=1)

        q2 = torch.zeros((R2.shape[0], 4), device=R.device, dtype=R.dtype)

        # idx == 0
        m0 = idx == 0
        if m0.any():
            S = torch.sqrt(1.0 + R2[m0, 0, 0] - R2[m0, 1, 1] - R2[m0, 2, 2]) * 2.0
            q2[m0, 0] = (R2[m0, 2, 1] - R2[m0, 1, 2]) / S
            q2[m0, 1] = 0.25 * S
            q2[m0, 2] = (R2[m0, 0, 1] + R2[m0, 1, 0]) / S
            q2[m0, 3] = (R2[m0, 0, 2] + R2[m0, 2, 0]) / S

        # idx == 1
        m1 = idx == 1
        if m1.any():
            S = torch.sqrt(1.0 + R2[m1, 1, 1] - R2[m1, 0, 0] - R2[m1, 2, 2]) * 2.0
            q2[m1, 0] = (R2[m1, 0, 2] - R2[m1, 2, 0]) / S
            q2[m1, 1] = (R2[m1, 0, 1] + R2[m1, 1, 0]) / S
            q2[m1, 2] = 0.25 * S
            q2[m1, 3] = (R2[m1, 1, 2] + R2[m1, 2, 1]) / S

        # idx == 2
        m2 = idx == 2
        if m2.any():
            S = torch.sqrt(1.0 + R2[m2, 2, 2] - R2[m2, 0, 0] - R2[m2, 1, 1]) * 2.0
            q2[m2, 0] = (R2[m2, 1, 0] - R2[m2, 0, 1]) / S
            q2[m2, 1] = (R2[m2, 0, 2] + R2[m2, 2, 0]) / S
            q2[m2, 2] = (R2[m2, 1, 2] + R2[m2, 2, 1]) / S
            q2[m2, 3] = 0.25 * S

        q[mask2] = q2

    # normalize
    q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-12)
    return q

def pose_from_T_world(T: np.ndarray, device: torch.device, dtype=torch.float32):
    """T: (4,4) numpy -> (pos_w (1,3), quat_wxyz_w (1,4)) torch"""
    pos = torch.tensor(T[0:3, 3], device=device, dtype=dtype).unsqueeze(0)
    R = torch.tensor(T[0:3, 0:3], device=device, dtype=dtype).unsqueeze(0)
    quat = quat_wxyz_from_rotmat(R)
    return pos, quat


def quat_angle_error(q_actual: np.ndarray, q_target: np.ndarray) -> float:
    """Compute angular error between two quaternions."""
    q_actual = q_actual / (np.linalg.norm(q_actual) + 1e-8)
    q_target = q_target / (np.linalg.norm(q_target) + 1e-8)
    dot_val = np.abs(np.dot(q_actual, q_target))
    return 2 * np.arccos(np.clip(dot_val, 0.0, 1.0))

def eulertorotmat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to rotation matrix."""
    rotmat = np.zeros((euler.shape[0], 3, 3))
    rotmat[:, 0, 0] = np.cos(euler[:, 1]) * np.cos(euler[:, 2])
    rotmat[:, 0, 1] = np.cos(euler[:, 1]) * np.sin(euler[:, 2]) * np.sin(euler[:, 0]) - np.sin(euler[:, 1]) * np.cos(euler[:, 0])
    rotmat[:, 0, 2] = np.cos(euler[:, 1]) * np.sin(euler[:, 2]) * np.cos(euler[:, 0]) + np.sin(euler[:, 1]) * np.sin(euler[:, 0])
    rotmat[:, 1, 0] = np.sin(euler[:, 1]) * np.cos(euler[:, 2])
    return rotmat

def rotmattoeuler(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles."""
    euler = np.zeros((rotmat.shape[0], 3))
    euler[:, 0] = np.arctan2(rotmat[:, 1, 2], rotmat[:, 2, 2])
    euler[:, 1] = np.arcsin(-rotmat[:, 0, 2])
    euler[:, 2] = np.arctan2(rotmat[:, 0, 1], rotmat[:, 0, 0])
    return euler

def eulertotquat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion(wxyz)"""
    euler = euler.reshape(-1, 3)
    quat = np.zeros((euler.shape[0], 4))
    # for each row, convert euler to quat
    quat[:, 0] = np.cos(euler[:, 0] / 2) * np.cos(euler[:, 1] / 2) * np.cos(euler[:, 2] / 2) + np.sin(euler[:, 0] / 2) * np.sin(euler[:, 1] / 2) * np.sin(euler[:, 2] / 2)
    quat[:, 1] = np.sin(euler[:, 0] / 2) * np.cos(euler[:, 1] / 2) * np.cos(euler[:, 2] / 2) - np.cos(euler[:, 0] / 2) * np.sin(euler[:, 1] / 2) * np.sin(euler[:, 2] / 2)
    quat[:, 2] = np.cos(euler[:, 0] / 2) * np.sin(euler[:, 1] / 2) * np.cos(euler[:, 2] / 2) + np.sin(euler[:, 0] / 2) * np.cos(euler[:, 1] / 2) * np.sin(euler[:, 2] / 2)
    quat[:, 3] = np.cos(euler[:, 0] / 2) * np.cos(euler[:, 1] / 2) * np.sin(euler[:, 2] / 2) - np.sin(euler[:, 0] / 2) * np.sin(euler[:, 1] / 2) * np.cos(euler[:, 2] / 2)
    return quat


def quattoeuler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles."""
    euler = np.zeros((quat.shape[0], 3))
    euler[:, 0] = np.arctan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2))
    euler[:, 1] = np.arcsin(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]))
    euler[:, 2] = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
    return euler