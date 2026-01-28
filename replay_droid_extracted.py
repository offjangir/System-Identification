#!/usr/bin/env python3
"""
replay_droid_extracted.py – Replay a single DROID extracted trajectory in Isaac Lab

This script loads a trajectory from extracted DROID format (scene_*/tcp.npy files),
replays it in Isaac Lab, and records a video.

Usage:
    ./isaaclab.sh -p scene_generation_isaaclab/replay_droid_extracted.py \
        --data_dir /path/to/droid_extracted \
        --scene_idx 0 \
        --out_dir ./results \
        --headless
"""

from __future__ import annotations


import os, sys

# Ensure this file's directory is on PYTHONPATH (Isaac python.sh sometimes doesn't include it)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

print("[DEBUG] CWD:", os.getcwd())
print("[DEBUG] THIS_DIR:", THIS_DIR)
print("[DEBUG] sys.path[0:5]:", sys.path[:5])

import argparse
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

# Import Isaac Lab app launcher first
from isaaclab.app import AppLauncher

# Add argument parser
parser = argparse.ArgumentParser(
    description="Replay extracted DROID trajectory with Isaac Lab simulator"
)
parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to directory containing scene_* subdirectories")
parser.add_argument("--scene_idx", type=int, default=1,
                    help="Scene index to replay (default: 0)")
parser.add_argument("--out_dir", type=str, default="./results_isaaclab",
                    help="Output directory for results")
parser.add_argument("--save_video", action="store_true", default=True,
                    help="Save video of trajectory")
parser.add_argument("--bad_pd", action="store_true",
                    help="Use worse PD parameters for comparison")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import rest of the modules
from transforms3d.affines import decompose44
from transforms3d.quaternions import mat2quat

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms
# from simulator_isaaclab import SimulatorIsaacLab

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# import data loader
from utilities.data_loader import load_droid_numpy_extracted_droid
from utilities.visualizer import spawn_trajectory_markers_usd_tcp, spawn_trajectory_markers_usd_cartesian
from utilities.rotations import eulertotquat, quattoeuler

# --------------------------------------------------------------------------- #
#  Scene Configuration                                                        #
# --------------------------------------------------------------------------- #
@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Scene configuration for trajectory replay."""

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Camera for rendering
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraSensor1",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            vertical_aperture=20.955,
            horizontal_aperture=20.955 * (640 / 480),
            clipping_range=(0.1, 1.0e3)
        )
    )


# --------------------------------------------------------------------------- #
# Main Replay Function                                                       #
# --------------------------------------------------------------------------- #
def main():
    """Replay DROID-extracted TCP trajectory with Diff-IK (pose targets)."""
    print(f"[INFO] Replaying scene {args_cli.scene_idx} from {args_cli.data_dir}")
    os.makedirs(args_cli.out_dir, exist_ok=True)
    # Load trajectory
    trajectories = load_droid_numpy_extracted_droid(args_cli.data_dir, n_envs=args_cli.scene_idx + 1)
    trajectory = trajectories[args_cli.scene_idx]
    cartesian_position = trajectory["cartesian_position"]  # (T,6)
    print(f"[INFO] Trajectory has {len(cartesian_position)} steps")

    # -----------------------------
    # Load trajectory and prepare initial Pose 
    # -----------------------------
    # trajectories = load_droid_extracted(args_cli.data_dir, n_envs=args_cli.scene_idx + 1)
    cartesian_position = trajectories[args_cli.scene_idx]["cartesian_position"]
    euler = cartesian_position[:, 3:6]
    quat = eulertotquat(euler)
    cartesian_position = np.concatenate([cartesian_position[:, :3], quat], axis=-1)
    
    cartesian_position = torch.from_numpy(cartesian_position).to(device="cuda:0", dtype=torch.float32)
    print(f"[INFO] cartesian_position: {cartesian_position.shape}")
    # -----------------------------
    # Sim + scene
    # -----------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    sim_utils.spawn_ground_plane("/World/defaultGroundPlane", cfg=sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light(
        "/World/Light",
        cfg=sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75)),
    )
    spawn_trajectory_markers_usd_cartesian(sim, cartesian_position, every_k=5, radius=0.01)

    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    robot = scene["robot"]
    camera = scene["camera"]

    sim.reset()
    scene.reset()

    print("[INFO] robot.joint_names:", robot.joint_names)
    print("[INFO] robot.body_names:", robot.body_names)

    robot_entity_cfg = SceneEntityCfg(
        name="robot",
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7"
                    ],
        body_names=["panda_hand"],
    )
    robot_entity_cfg.resolve(scene)
    # -----------------------------
    # Viewer camera pose (optional)
    # -----------------------------
    camera_positions = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
    camera_targets = torch.tensor([[2.0, 2.0, 2.0]], device=sim.device, dtype=torch.float32)
    camera.set_world_poses_from_view(camera_targets, camera_positions)

    print("[INFO] Scene and simulation reset complete")
    # -----------------------------
    # Warm-up step
    # -----------------------------
    # One warm-up step so buffers populate
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    camera.update(sim.get_physics_dt())

    # CRITICAL: Set joint position targets to current positions to prevent falling
    robot.set_joint_position_target(robot.data.joint_pos.clone())
    # -----------------------------
    # Resolve EE + joints
    # -----------------------------
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1  # PhysX jacobian indexing
    joint_ids = robot_entity_cfg.joint_ids

    print("[INFO] joint_ids:", joint_ids)
    print("[INFO] ee_body_id:", ee_body_id, " ee_jacobi_idx:", ee_jacobi_idx)
    
    # -----------------------------
    # Diff-IK controller
    # -----------------------------
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    ik.reset()

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos7 = joint_pos[:, :7]
    joint_vel7 = joint_vel[:, :7]
    robot.write_joint_state_to_sim(joint_pos7, joint_vel7, joint_ids=joint_ids)
    robot.reset()
    ik.reset()


    # -----------------------------
    # Warm-up step
    # -----------------------------
    # One warm-up step so buffers populate
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    camera.update(sim.get_physics_dt())

    # CRITICAL: Set joint position targets to current positions to prevent falling
    robot.set_joint_position_target(robot.data.joint_pos.clone())

    # -----------------------------
    # Initialize robot from first TCP pose
    # -----------------------------
    print("[INFO] Initializing robot from first TCP frame")

    print("[INFO] robot.data.joint_pos:", robot.data.joint_pos)
    # Target EE pose in WORLD
    # target_pos_w, target_quat_w = pose_from_T_world(tcp_T[0], sim.device)
    target_pos_w = cartesian_position[0, :3]
    target_quat_w = cartesian_position[0, 3:7]
    print("[INFO] target_pos_w:", target_pos_w.shape)
    print("[INFO] target_quat_w:", target_quat_w.shape)
    target_pos_w = target_pos_w.view(1, 3).repeat(scene.num_envs, 1)
    target_quat_w = target_quat_w.view(1, 4).repeat(scene.num_envs, 1)

    root_pose_w = robot.data.root_pose_w 
    print("[INFO] root_pos_w:", root_pose_w[:, 0:3].shape)
    print("[INFO] robot_quat_w:", root_pose_w[:, 3:7].shape)

    # --- convert WORLD target -> BASE target (because we measure EE in base below) ---
    root_pose_w = robot.data.root_pose_w  # (N,7)
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        target_pos_w.repeat(scene.num_envs, 1),
        target_quat_w.repeat(scene.num_envs, 1),
    )
    print("[INFO] target_pos_b:", target_pos_b)
    print("[INFO] target_quat_b:", target_quat_b)

    # Set IK command (base frame)
    ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))  # (N,7)

    # Read current EE pose, compute base-frame EE, jacobian, joints
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]  # (N,7)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )

    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]  # (N,6,7)
    joint_pos = robot.data.joint_pos[:, joint_ids]  # (N,7)

    # IK -> desired arm joints
    des_q = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)  # (N,7)
    # set directly to the desired joint positions
    # make joint vel sam shape as des_q  by removing last joint elements on basis of joint_ids
    joint_vel7 = joint_vel[:, :7]
    robot.write_joint_state_to_sim(des_q, joint_vel7, joint_ids=joint_ids)
    robot.reset()

    # -----------------------------
    # Warm-up step
    # -----------------------------
    # One warm-up step so buffers populate
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    camera.update(sim.get_physics_dt())

    # CRITICAL: Set joint position targets to current positions to prevent falling
    robot.set_joint_position_target(robot.data.joint_pos.clone())

    print("[INFO] joint_pos:", robot.data.joint_pos)
    print("[INFO] Initialization to first TCP pose complete")
    # -----------------------------
    # Target marker (world)
    # -----------------------------
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    target_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/target"))

    # -----------------------------
    # Tracking loop
    # -----------------------------
    substeps = getattr(args_cli, "substeps", 10)  # physics steps per waypoint
    save_video = bool(getattr(args_cli, "save_video", True))

    frames = []

    import imageio
    gif_path = os.path.join(args_cli.out_dir, f"tcp_track_scene{args_cli.scene_idx}.gif")
    writer = imageio.get_writer(gif_path, mode="I", fps=20)

    print(f"[INFO] Writing GIF to {gif_path}")

    for step in range(len(cartesian_position)):
        # --- target pose in WORLD from 4x4 ---
        # (expects helper to return (1,3) and (1,4) wxyz on sim.device)
        # target_pos_w, target_quat_w = pose_from_T_world(tcp_T[step], sim.device)

        target_pos_w = cartesian_position[step, :3]
        target_quat_w = cartesian_position[step, 3:7]
        if isinstance(target_pos_w, np.ndarray):
            target_pos_w = torch.from_numpy(target_pos_w).float().to(scene.env_origins.device)
        if isinstance(target_quat_w, np.ndarray):
            target_quat_w = torch.from_numpy(target_quat_w).float().to(scene.env_origins.device)

        target_pos_w = target_pos_w.view(1, 3).repeat(scene.num_envs, 1)
        target_quat_w = target_quat_w.view(1, 4).repeat(scene.num_envs, 1)
        # Visualize target (marker wants world positions; add env origin)
        target_marker.visualize(target_pos_w + scene.env_origins, target_quat_w)

        # --- convert WORLD target -> BASE target (because we measure EE in base below) ---
        root_pose_w = robot.data.root_pose_w  # (N,7)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            target_pos_w.repeat(scene.num_envs, 1),
            target_quat_w.repeat(scene.num_envs, 1),
        )

        # Set IK command (base frame)
        ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))  # (N,7)

        # Read current EE pose, compute base-frame EE, jacobian, joints
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]  # (N,7)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]  # (N,6,7)
        joint_pos = robot.data.joint_pos[:, joint_ids]  # (N,7)

        # IK -> desired arm joints
        des_q = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)  # (N,7)

        # Apply for a few substeps so it moves toward the target
        for _ in range(substeps):
            root_pose_w = robot.data.root_pose_w
            ee_pose_w = robot.data.body_pose_w[:, ee_body_id]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],   ee_pose_w[:, 3:7],
            )

            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]
            joint_pos = robot.data.joint_pos[:, joint_ids]

            des_q = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(des_q, joint_ids=joint_ids)

            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt())
            frame = camera.data.output["rgb"][0].detach().cpu().numpy()
            writer.append_data(frame)
            frames.append(frame)

        # quick world position error (debug)
        ee_pose_w_now = robot.data.body_pose_w[:, ee_body_id]
        ee_pos_w_now = ee_pose_w_now[0, 0:3]
        pos_err = torch.norm(
                ee_pos_w_now - torch.tensor(cartesian_position[step, :3],
                                            device=ee_pos_w_now.device,
                                            dtype=torch.float32)
            ).item()
        print(f"[step {step+1}/{len(cartesian_position)}] pos_err_w = {pos_err:.4f}")

    # -----------------------------
    # Save video
    # -----------------------------
    if save_video and len(frames) > 0:
        video_path = os.path.join(args_cli.out_dir, f"tcp_track_scene{args_cli.scene_idx}.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"[INFO] Saved video to {video_path} ({len(frames)} frames)")

if __name__ == "__main__":
    main()
    simulation_app.close()
