import os
import numpy as np
import glob
import pickle

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


# --------------------------------------------------------------------------- #
#  Bridge Dataset Loading                                                      #
# --------------------------------------------------------------------------- #

def load_bridge_dataset(root_dir: str, max_trajectories: int = None):
    """
    Load trajectories from Bridge dataset format (raw pickle files).
    
    The Bridge dataset structure:
    root_dir/
        toykitchen2/ or toykitchen7/
            task_name/
                episode_id/
                    timestamp/
                        raw/
                            traj_group0/
                                trajN/
                                    obs_dict.pkl
                                    
    obs_dict.pkl contains:
        - eef_transform: (T, 4, 4) end-effector transformation matrices
        - qpos: (T, 6) joint positions for WidowX (6 DOF arm)
        - qvel: (T, 6) joint velocities
        - state: (T, 7) state [x, y, z, qx, qy, qz, gripper]
                                    
    Args:
        root_dir: Path to the bridge dataset root (e.g., /path/to/bridge/raw/rss)
        max_trajectories: Maximum number of trajectories to load (None for all)
        
    Returns:
        List of trajectory dictionaries with keys:
        - cartesian_position: (T, 6) array of [x, y, z, roll, pitch, yaw]
        - qpos: (T, 6) joint positions
        - eef_transform: (T, 4, 4) end-effector transforms
        - path: original file path
    """
    from transforms3d.euler import mat2euler
    
    trajectories = []
    
    # Find all obs_dict.pkl files recursively
    #  Load only 100 obs_dict.pkl files
    obs_dict_files = []
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if "obs_dict.pkl" in files:
            obs_dict_files.append(os.path.join(root, "obs_dict.pkl"))
            count += 1
            if count >= 100:
                break

    obs_dict_files = sorted(obs_dict_files)
    
    if len(obs_dict_files) == 0:
        raise RuntimeError(f"No obs_dict.pkl files found in {root_dir}")
    
    print(f"[INFO] Found {len(obs_dict_files)} trajectories in {root_dir}")
    
    if max_trajectories is not None:
        obs_dict_files = obs_dict_files[:max_trajectories]
    
    for obs_path in obs_dict_files:
        try:
            with open(obs_path, "rb") as f:
                obs_dict = pickle.load(f)
            
            # Extract end-effector transforms (T, 4, 4)
            eef_transform = obs_dict["eef_transform"].astype(np.float64)
            
            # Convert 4x4 transforms to [x, y, z, roll, pitch, yaw]
            T = eef_transform.shape[0]
            cartesian_position = np.zeros((T, 6), dtype=np.float64)
            
            for i in range(T):
                # Position from translation
                cartesian_position[i, :3] = eef_transform[i, :3, 3]
                # Euler angles from rotation matrix (using sxyz convention)
                euler = mat2euler(eef_transform[i, :3, :3], axes='sxyz')
                cartesian_position[i, 3:6] = euler
            
            # Extract joint positions (T, 6) for WidowX
            qpos = obs_dict["qpos"].astype(np.float64)
            
            trajectories.append({
                "cartesian_position": cartesian_position,
                "qpos": qpos,
                "eef_transform": eef_transform,
                "path": obs_path,
            })
            
        except Exception as e:
            print(f"[WARN] Failed to load {obs_path}: {e}")
            continue
    
    print(f"[INFO] Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def load_bridge_dataset_numpy_extracted(root_dir: str, max_trajectories: int = None):
    """
    Load trajectories from Bridge dataset that has been pre-extracted to numpy format.
    
    Expected format (similar to DROID extracted):
    root_dir/
        episode_*_cartesian_position.npy  - (T, 6) [x, y, z, roll, pitch, yaw]
        episode_*_qpos.npy                - (T, 6) joint positions
        
    Args:
        root_dir: Path to extracted numpy files
        max_trajectories: Maximum number of trajectories to load
        
    Returns:
        List of trajectory dictionaries
    """
    traj_files = sorted(
        glob.glob(os.path.join(root_dir, "episode_*_cartesian_position.npy"))
    )
    
    if len(traj_files) == 0:
        raise RuntimeError(f"No trajectory files found in {root_dir}")
    
    if max_trajectories is not None:
        traj_files = traj_files[:max_trajectories]
    
    trajectories = []
    for path in traj_files:
        cartesian_position = np.load(path).astype(np.float64)
        
        # Try to load corresponding qpos file
        qpos_path = path.replace("_cartesian_position.npy", "_qpos.npy")
        qpos = None
        if os.path.exists(qpos_path):
            qpos = np.load(qpos_path).astype(np.float64)
        
        traj_dict = {"cartesian_position": cartesian_position}
        if qpos is not None:
            traj_dict["qpos"] = qpos
            
        trajectories.append(traj_dict)
    
    return trajectories


def load_data_from_bridge_data(root_dir: str):
    """
    Load data from Bridge data format.
    """
    data = np.load(os.path.join(root_dir, "data.npy"))
    return data


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
