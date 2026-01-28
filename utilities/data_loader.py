import os
import numpy as np
import glob

# --------------------------------------------------------------------------- #
#  Data Loading                                                               #
# --------------------------------------------------------------------------- #

def load_droid_numpy_extracted_droid(root_dir: str):
    """
    Load ALL trajectories from a DROID numpy directory.
    """
    traj_files = sorted(
        glob.glob(os.path.join(root_dir, "episode_*_cartesian_position.npy"))
    )

    if len(traj_files) == 0:
        raise RuntimeError(f"No trajectory files found in {root_dir}")

    trajectories = []
    for path in traj_files:
        cartesian_position = np.load(path).astype(np.float64)
        trajectories.append({
            "cartesian_position": cartesian_position
        })

    return trajectories

def load_droid_extracted_droid(root_dir: str, n_envs: int):
    """
    Load trajectories from DROID numpy format.
    """
    trajectories = []

    print(f"[INFO] Loading {n_envs} scenes from {root_dir}")

    for i in range(n_envs):
        scene_dir = os.path.join(root_dir, f"scene_{i}")
        tcp_path = os.path.join(scene_dir, "tcp.npy")

        if not os.path.isfile(tcp_path):
            print(f"[WARN] Missing {tcp_path}, skipping")
            continue

        tcp = np.load(tcp_path).astype(np.float64)
        good = np.isfinite(tcp).all(axis=(1, 2))
        tcp = tcp[good]
        #  Add 0.5 in x in all the tcp frames
        tcp[:, 0, 3] += 0.5
        
        trajectories.append({
            "tcp_T": tcp,
            "scene_dir": scene_dir
        })

    if not trajectories:
        raise RuntimeError(f"No valid scene_i/tcp.npy found under {root_dir}")

    print(f"[INFO] Loaded {len(trajectories)} scenes")
    return trajectories




