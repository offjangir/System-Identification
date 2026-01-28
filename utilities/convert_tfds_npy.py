#!/usr/bin/env python3
"""
convert_tfds_npy.py – Convert TensorFlow Datasets (RLDS) to NumPy format

This script loads episodes from a TensorFlow Datasets RLDS directory and converts
them to NumPy arrays for easier loading without TensorFlow dependencies.

Usage:
    python helper/convert_tfds_npy.py \
        --dataset_dir /data/user_data/yjangir/yash/IsaacLab-Cluster/scene_generation_isaaclab/data/droid_100/1.0.0/ \
        --out_dir /data/user_data/yjangir/yash/IsaacLab-Cluster/scene_generation_isaaclab/data/droid_numpy/ \
        --n_episodes 100 \
        --save_joints \
        --save_rewards
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("[ERROR] tensorflow_datasets is required. Install with: pip install tensorflow tensorflow-datasets")
    sys.exit(1)

import tensorflow as tf

def _to_np(x):
    """tf.Tensor -> np.ndarray, pass through np arrays."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)

def _stack_steps(steps_ds, key_path):
    """
    steps_ds: tf.data.Dataset yielding step dicts
    key_path: tuple of keys to descend, e.g. ("observation","cartesian_position")
    returns: np.ndarray stacked over time [T, ...]
    """
    out = []
    for step in steps_ds:
        v = step
        for k in key_path:
            v = v[k]
        out.append(_to_np(v))
    if len(out) == 0:
        raise ValueError(f"No steps found for key_path={key_path}")
    return np.stack(out, axis=0)


def load_rlds_builder(dataset_dir: str):
    """
    Load RLDS dataset builder from directory.
    
    Args:
        dataset_dir: Path to directory containing TFDS builder files
        
    Returns:
        TensorFlow Datasets builder object
        
    Raises:
        ValueError: If dataset directory is invalid
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset path is not a directory: {dataset_dir}")
    
    try:
        builder = tfds.builder_from_directory(dataset_dir)
        return builder
    except Exception as e:
        raise ValueError(f"Could not load dataset from {dataset_dir}: {e}")


def convert_episode_to_numpy(
    episode: dict,
    episode_idx: int,
    output_dir: str,
    save_joints: bool = False,
    save_rewards: bool = False,
    save_actions: bool = False,
) -> dict:
    """
    Convert a single episode to NumPy arrays and save to disk.
    
    Args:
        episode: Episode dictionary from TFDS
        episode_idx: Episode index for naming files
        output_dir: Directory to save NumPy files
        save_joints: Whether to save joint positions
        save_rewards: Whether to save rewards
        save_actions: Whether to save actions
        
    Returns:
        Dictionary with saved file paths and statistics
    """
    steps = episode["steps"]
    saved_files = {}
    stats = {}
    
    # Extract cartesian position (required)
    if "cartesian_position" not in steps["observation"]:
        raise ValueError(f"Episode {episode_idx} missing 'cartesian_position' in observations")
    
    cartesian_pos = steps["observation"]["cartesian_position"].astype(np.float64)
    num_steps = cartesian_pos.shape[0]
    
    # Save cartesian position
    cart_file = os.path.join(output_dir, f"episode_{episode_idx:05d}_cartesian_position.npy")
    np.save(cart_file, cartesian_pos)
    saved_files["cartesian_position"] = cart_file
    stats["num_steps"] = num_steps
    stats["cartesian_shape"] = cartesian_pos.shape
    
    # Save joint positions (optional)
    if save_joints:
        if "joint_position" in steps["observation"]:
            joint_pos = steps["observation"]["joint_position"].astype(np.float64)
            joint_file = os.path.join(output_dir, f"episode_{episode_idx:05d}_joint_position.npy")
            np.save(joint_file, joint_pos)
            saved_files["joint_position"] = joint_file
            stats["joint_shape"] = joint_pos.shape
        else:
            print(f"[WARN] Episode {episode_idx} missing 'joint_position' in observations")
    
    # Save rewards (optional)
    if save_rewards:
        reward = steps.get("reward", None)
        if reward is not None:
            reward_file = os.path.join(output_dir, f"episode_{episode_idx:05d}_reward.npy")
            np.save(reward_file, reward.astype(np.float32))
            saved_files["reward"] = reward_file
            stats["reward_range"] = (float(reward.min()), float(reward.max()))
        else:
            print(f"[WARN] Episode {episode_idx} missing 'reward' field")
    
    # Save actions (optional)
    if save_actions:
        if "action" in steps:
            action = steps["action"].astype(np.float32)
            action_file = os.path.join(output_dir, f"episode_{episode_idx:05d}_action.npy")
            np.save(action_file, action)
            saved_files["action"] = action_file
            stats["action_shape"] = action.shape
        else:
            print(f"[WARN] Episode {episode_idx} missing 'action' field")
    
    return {"files": saved_files, "stats": stats}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow Datasets (RLDS) episodes to NumPy format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert 100 episodes with cartesian positions only
  python convert_tfds_npy.py --dataset_dir /path/to/droid_100 --out_dir ./output --n_episodes 100
  
  # Convert with joints and rewards
  python convert_tfds_npy.py --dataset_dir /path/to/droid_100 --out_dir ./output \\
      --n_episodes 100 --save_joints --save_rewards
  
  # Convert all available episodes
  python convert_tfds_npy.py --dataset_dir /path/to/droid_100 --out_dir ./output --n_episodes -1
        """
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to TensorFlow Datasets directory (contains dataset_info.json)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for NumPy files"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of episodes to convert (default: 100). Use -1 for all available"
    )
    parser.add_argument(
        "--save_joints",
        action="store_true",
        help="Save joint positions"
    )
    parser.add_argument(
        "--save_rewards",
        action="store_true",
        help="Save reward signals"
    )
    parser.add_argument(
        "--save_actions",
        action="store_true",
        help="Save action commands"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting episode index (for resuming, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset_dir):
        print(f"[ERROR] Dataset directory does not exist: {args.dataset_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.out_dir}")
    
    # Load dataset
    print(f"[INFO] Loading dataset from: {args.dataset_dir}")
    try:
        builder = load_rlds_builder(args.dataset_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)
    
    # Get dataset info
    try:
        dataset_info = builder.info
        total_episodes = dataset_info.splits["train"].num_examples
        print(f"[INFO] Dataset contains {total_episodes} episodes")
    except Exception as e:
        print(f"[WARN] Could not get dataset info: {e}")
        total_episodes = None
    
    # Determine number of episodes to process
    if args.n_episodes == -1:
        if total_episodes is not None:
            num_episodes = total_episodes - args.start_idx
        else:
            print("[ERROR] Cannot determine total episodes. Please specify --n_episodes")
            sys.exit(1)
    else:
        num_episodes = args.n_episodes
    
    if args.start_idx > 0:
        print(f"[INFO] Starting from episode {args.start_idx}")
    
    print(f"[INFO] Converting {num_episodes} episodes")
    print(f"[INFO] Saving: cartesian_position (required)")
    if args.save_joints:
        print(f"[INFO] Saving: joint_position")
    if args.save_rewards:
        print(f"[INFO] Saving: reward")
    if args.save_actions:
        print(f"[INFO] Saving: action")
    
    # Load dataset
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    # Skip to start index
    if args.start_idx > 0:
        ds = ds.skip(args.start_idx)
    
        # Process episodes (RLDS: episode["steps"] is a tf.data.Dataset)
    converted = 0
    failed = 0
    total_steps = 0

    try:
        for ep_idx, episode in enumerate(tqdm(ds.take(num_episodes), desc="Converting", total=num_episodes)):
            actual_idx = args.start_idx + ep_idx
            try:
                steps_ds = episode["steps"]  # DatasetV2 of per-step dicts

                # Required: cartesian_position -> [T, 6]
                cartesian_pos = _stack_steps(steps_ds, ("observation", "cartesian_position")).astype(np.float64)
                num_steps = cartesian_pos.shape[0]

                cart_file = os.path.join(args.out_dir, f"episode_{actual_idx:05d}_cartesian_position.npy")
                np.save(cart_file, cartesian_pos)

                # Optional: joint_position -> [T, 7]
                if args.save_joints:
                    try:
                        joint_pos = _stack_steps(steps_ds, ("observation", "joint_position")).astype(np.float64)
                        joint_file = os.path.join(args.out_dir, f"episode_{actual_idx:05d}_joint_position.npy")
                        np.save(joint_file, joint_pos)
                    except Exception as e:
                        print(f"\n[WARN] Episode {actual_idx} joint_position not saved: {e}")

                # Optional: reward -> [T]
                if args.save_rewards:
                    try:
                        reward = _stack_steps(steps_ds, ("reward",)).astype(np.float32)
                        reward_file = os.path.join(args.out_dir, f"episode_{actual_idx:05d}_reward.npy")
                        np.save(reward_file, reward)
                    except Exception as e:
                        print(f"\n[WARN] Episode {actual_idx} reward not saved: {e}")

                # Optional: action -> [T, ...]
                if args.save_actions:
                    try:
                        action = _stack_steps(steps_ds, ("action",)).astype(np.float32)
                        action_file = os.path.join(args.out_dir, f"episode_{actual_idx:05d}_action.npy")
                        np.save(action_file, action)
                    except Exception as e:
                        print(f"\n[WARN] Episode {actual_idx} action not saved: {e}")

                converted += 1
                total_steps += num_steps

            except Exception as e:
                print(f"\n[ERROR] Failed to convert episode {actual_idx}: {e}")
                failed += 1
                continue

    except KeyboardInterrupt:
        print("\n[INFO] Conversion interrupted by user")

    
    # Print summary
    print(f"\n[INFO] Conversion complete!")
    print(f"[INFO] Successfully converted: {converted} episodes")
    print(f"[INFO] Failed: {failed} episodes")
    if converted > 0:
        print(f"[INFO] Total steps: {total_steps}")
        print(f"[INFO] Average steps per episode: {total_steps / converted:.1f}")
    print(f"[INFO] Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
