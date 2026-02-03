# Copyright (c) 2024-2025, Yash Jangir
# Isaac Lab port of the Genesis multi-environment simulator for system identification
#
# This module provides a multi-environment simulator wrapper for Isaac Lab,
# designed for system identification tasks with the Franka Panda robot.


from __future__ import annotations


import os, sys
import argparse
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

# Import Isaac Lab app launcher first
from isaaclab.app import AppLauncher

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

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG,FRANKA_PANDA_CFG
import isaaclab_assets as isaaclab_assets
print([x for x in dir(isaaclab_assets) if "FRANKA" in x])
# import data loader
from utilities.data_loader import load_droid_numpy_extracted_droid
from utilities.visualizer import spawn_trajectory_markers_usd_tcp, spawn_trajectory_markers_usd_cartesian
from utilities.rotations import eulertotquat, quattoeuler
from isaaclab.actuators import ImplicitActuator


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

# @configclass
# class PandaSysIDSceneCfg(InteractiveSceneCfg):
#     """Scene configuration for Panda system identification."""
#     robot = PandaConfig.replace(prim_path="{ENV_REGEX_NS}/Robot")
#     # Camera for rendering
#     camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/CameraSensor1",
#         update_period=0,
#         height=480,
#         width=640,
#         data_types=["rgb"],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             vertical_aperture=20.955,
#             horizontal_aperture=20.955 * (640 / 480),
#             clipping_range=(0.1, 1.0e3)
#         )
#     )

# @configclass
# class UR5SysIDSceneCfg(InteractiveSceneCfg):
#     """Scene configuration for UR5 system identification."""
#     robot = UR5Config.replace(prim_path="{ENV_REGEX_NS}/Robot")
#     # Camera for rendering
#     camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/CameraSensor1",
#         update_period=0,
#         height=480,
#         width=640,
#         data_types=["rgb"],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             vertical_aperture=20.955,
#             horizontal_aperture=20.955 * (640 / 480),
#             clipping_range=(0.1, 1.0e3)
#         )
#     )

# @configclass
# class WidowXSysIDSceneCfg(InteractiveSceneCfg):
#     """Scene configuration for WidowX system identification."""
#     robot = WidowXConfig.replace(prim_path="{ENV_REGEX_NS}/Robot")
#     # Camera for rendering
#     camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/CameraSensor1",
#         update_period=0,
#         height=480,
#         width=640,
#         data_types=["rgb"],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             vertical_aperture=20.955,   
#             horizontal_aperture=20.955 * (640 / 480),
#             clipping_range=(0.1, 1.0e3)
#         )
#     )

class SimulatorIsaacLab:
    """
    A multi-environment simulator wrapper for Isaac Lab.
    
    This class manages:
     - Scene creation with parallel environments
     - Robot control (position, velocity, force/torque)
     - Differential IK for task-space control
     - State retrieval (joint positions, velocities, end-effector pose)
    """

    def __init__(
        self,
        robot_name: str = "franka",
        n_envs: int = 1,
        device: Optional[str] = None,
        show_viewer: bool = False,
        dt: float = 0.01,
        env_spacing: float = 2.0,
        record: bool = False,
    ):
        """
        Initialize the Isaac Lab simulator wrapper.
        
        Args:
            robot_name: Name of the robot (currently only 'franka' supported)
            n_envs: Number of parallel environments
            device: Device to run simulation on ('cuda:0', 'cpu', etc.)
            show_viewer: Whether to show the GUI viewer
            dt: Simulation timestep
            env_spacing: Spacing between environments in the grid
        """
        global record_demo 
        self.record_demo = record

        self.robot_name = robot_name
        self.n_envs = n_envs
        self.device = device if device is not None else "cuda:0"
        self.show_viewer = show_viewer
        self.dt = dt
        self.env_spacing = env_spacing
        self.sim = None
        self.scene = None
        self.robot = None
        self.diff_ik_controller = None
        
        # Robot-specific parameters
        self.ee_link_name = "panda_hand"
        self.arm_joint_names = ["panda_joint.*"]
        self.n_arm_joints = 7
        self.n_finger_joints = 2
        self.n_total_joints = 9


    def start_sim(self, args_cli=None):
        """
        Initialize the simulation, create scene, and add entities.
        
        Args:
            args_cli: Command line arguments (passed from AppLauncher)
        """

        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.dt,
            device=self.device        
            )

        self.sim = sim_utils.SimulationContext(sim_cfg)
        sim_utils.spawn_ground_plane("/World/defaultGroundPlane", cfg=sim_utils.GroundPlaneCfg())
        sim_utils.spawn_light(
            "/World/Light",
            cfg=sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75)),
        )
        # Set camera view
        if self.show_viewer:
            self.sim.set_camera_view([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])
        
        print(f"[INFO] Setting up robot config for {self.robot_name}")
        scene_cfg = self.setup_robot_config(self.robot_name)


        # Create the interactive scene
        self.scene = InteractiveScene(scene_cfg)
        
        # Get robot reference
        self.robot = self.scene["robot"]
        # if self.record_demo == True:
        self.camera = self.scene["camera"]


        self.sim.reset()
        self.scene.reset()

        # Setup robot entity configuration for IK
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["panda_joint.*"],
            body_names=[self.ee_link_name],
        )
        self.robot_entity_cfg.resolve(self.scene)

        # Get end-effector Jacobian index
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # -----------------------------
        # Resolve EE + joints
        self.ee_body_id = self.robot_entity_cfg.body_ids[0]
        self.ee_jacobi_idx = self.ee_body_id - 1  # PhysX jacobian indexing
        self.joint_ids = self.robot_entity_cfg.joint_ids

        # Initialize differential IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg,
            num_envs=self.n_envs,
            device=self.device,
        )

        self.warmup()
        # if self.record_demo == True:
        self.set_cam_pose()


        # Reset simulation
        self.sim.reset()
        
        self.warmup()

        # CRITICAL: Set joint position targets to current positions to prevent falling
        # Without this, the implicit actuators have no target and the arm falls under gravity
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())
        
        #  Use default joint positions for warmup
        default_joint_pos = self.robot.data.default_joint_pos.clone()[:,:self.ee_jacobi_idx]
        default_joint_vel = self.robot.data.default_joint_vel.clone()[:,:self.ee_jacobi_idx]
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, joint_ids=self.joint_ids)

        #  Reset the robot
        self.robot.reset()
        self.diff_ik_controller.reset()

        # Additional warm-up steps with position targets maintained
        for _ in range(5):
            self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.dt)
        
        print(f"[INFO] Isaac Lab simulator initialized with {self.n_envs} environments")

    def set_cam_pose(self, camera_targets: torch.Tensor, camera_positions: torch.Tensor):
        """
        Set the camera pose.
        """
        self.camera_targets = camera_targets
        self.camera_positions = camera_positions
        self.camera.set_world_poses_from_view(self.camera_targets, self.camera_positions)
        self.camera.update(self.dt)
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(self.dt)
        self.camera.update(self.dt)
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())


    def set_pd_all(self, kp, kd, env_ids=None):
        robot = self.scene["robot"]
        device = self.device

        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=device)
        env_ids = env_ids.to(device=device, dtype=torch.long)

        kp_t = torch.as_tensor(kp, device=device, dtype=torch.float32)
        kd_t = torch.as_tensor(kd, device=device, dtype=torch.float32)

        # normalize to (n_envs, num_joints)
        if kp_t.dim() == 1:
            kp_t = kp_t.unsqueeze(0).expand(self.n_envs, -1)
        if kd_t.dim() == 1:
            kd_t = kd_t.unsqueeze(0).expand(self.n_envs, -1)

        # robust num joints
        if hasattr(robot, "num_joints"):
            num_joints = robot.num_joints
        else:
            num_joints = int(robot.data.joint_pos.shape[1])

        if kp_t.shape[1] != num_joints or kd_t.shape[1] != num_joints:
            raise ValueError(f"kp/kd must have {num_joints} entries. Got {kp_t.shape}, {kd_t.shape}")

        joint_ids = torch.arange(num_joints, device=device, dtype=torch.long)

        # IMPORTANT: write to sim through the articulation/asset
        # Depending on IsaacLab version, this is either robot.write_* or robot.asset.write_*
        if hasattr(robot, "write_joint_stiffness_to_sim"):
            robot.write_joint_stiffness_to_sim(kp_t[env_ids, :], joint_ids=joint_ids, env_ids=env_ids)
            robot.write_joint_damping_to_sim(kd_t[env_ids, :], joint_ids=joint_ids, env_ids=env_ids)
        elif hasattr(robot, "asset") and hasattr(robot.asset, "write_joint_stiffness_to_sim"):
            robot.asset.write_joint_stiffness_to_sim(kp_t[env_ids, :], joint_ids=joint_ids, env_ids=env_ids)
            robot.asset.write_joint_damping_to_sim(kd_t[env_ids, :], joint_ids=joint_ids, env_ids=env_ids)
        else:
            raise AttributeError("Cannot find write_joint_stiffness_to_sim / write_joint_damping_to_sim on robot or robot.asset.")

    def setup_robot_config(self, robot_name: str):
        """
        Setup the robot configuration.
        """
        if robot_name == "franka":
            return ReplaySceneCfg(num_envs=self.n_envs, env_spacing=2.0, replicate_physics=True)
        else:
            raise ValueError(f"Invalid robot name: {robot_name}")

    def warmup(self):
        """
        Warm-up the simulation.
        """
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())

    def set_cam_pose(self):
        """
        Set the camera pose.
        """
        self.camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        self.camera_positions = torch.tensor([[2.0, 2.0, 2.0]], device=self.device, dtype=torch.float32)
        self.camera_targets = self.camera_targets.repeat(self.n_envs, 1)
        self.camera_positions = self.camera_positions.repeat(self.n_envs, 1)
        self.camera.set_world_poses_from_view(self.camera_positions, self.camera_targets)
        self.camera.update(self.dt)

    def step(self, render: bool = False, steps: int = 1):
        """
        Step the simulation forward.
        
        Args:
            steps: Number of simulation steps to take
        """
        for _ in range(steps):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.dt)

    # ---------- State Retrieval ----------
    def get_qpos(self) -> torch.Tensor:
        """
        Get current joint positions.
        
        Returns:
            Joint positions tensor of shape (n_envs, n_total_joints)
        """
        return self.robot.data.joint_pos.clone()

    def get_qvel(self) -> torch.Tensor:
        """
        Get current joint velocities.
        
        Returns:
            Joint velocities tensor of shape (n_envs, n_total_joints)
        """
        return self.robot.data.joint_vel.clone()

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get end-effector pose in world frame.
        
        Returns:
            Tuple of (position, quaternion) tensors:
            - position: shape (n_envs, 3)
            - quaternion: shape (n_envs, 4) in wxyz format
        """
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]  # (N,7)
        ee_pos = ee_pose_w[:, 0:3]
        ee_quat = ee_pose_w[:, 3:7]  # wxyz format
        return ee_pos, ee_quat

    def get_jacobian(self) -> torch.Tensor:
        """
        Get the geometric Jacobian for the end-effector.
        
        Returns:
            Jacobian tensor of shape (n_envs, 6, n_arm_joints)
        """
        jacobian = self.robot.root_physx_view.get_jacobians()
        return jacobian[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]

    # ---------- Control Methods ----------

    def set_qpos(self, qpos: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        Set joint positions directly (teleport).
        
        Args:
            qpos: Desired joint positions of shape (n_envs, n_joints) or (n_joints,)
            env_ids: Optional environment indices to set
        """
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0).expand(self.n_envs, -1)
        
        joint_vel = torch.zeros_like(qpos)
        
        if env_ids is not None:
            self.robot.write_joint_state_to_sim(qpos, joint_vel, env_ids=env_ids)
            # Set joint position targets to maintain position after teleport
            self.robot.set_joint_position_target(qpos, env_ids=env_ids)
        else:
            self.robot.write_joint_state_to_sim(qpos, joint_vel)
            # Set joint position targets to maintain position after teleport
            self.robot.set_joint_position_target(qpos)

    def control_position(self, qpos_target: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        Set joint position targets for PD control.
        
        Args:
            qpos_target: Target joint positions of shape (n_envs, n_joints) or (n_joints,)
            env_ids: Optional environment indices to control
        """
        if qpos_target.dim() == 1:
            qpos_target = qpos_target.unsqueeze(0).expand(self.n_envs, -1)
        
        if env_ids is not None:
            self.robot.set_joint_position_target(qpos_target, env_ids=env_ids)
        else:
            self.robot.set_joint_position_target(qpos_target)

    def control_force(self, torques: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        Apply joint torques directly.
        
        Args:
            torques: Joint torques of shape (n_envs, n_joints) or (n_joints,)
            env_ids: Optional environment indices to control
        """
        if torques.dim() == 1:
            torques = torques.unsqueeze(0).expand(self.n_envs, -1)
        
        if env_ids is not None:
            self.robot.set_joint_effort_target(torques, env_ids=env_ids)
        else:
            self.robot.set_joint_effort_target(torques)

    def set_pd_gains(self, kp: np.ndarray, kd: np.ndarray):
        """
        Set PD gains for the robot joints.
        
        Note: In Isaac Lab, PD gains are typically set via the actuator configuration.
        This method updates the internal stiffness and damping values.
        
        Args:
            kp: Position gains array of shape (n_joints,)
            kd: Velocity/damping gains array of shape (n_joints,)
        """
        # Store gains for manual PD computation
        self.kp = torch.tensor(kp, device=self.device, dtype=torch.float32)
        self.kd = torch.tensor(kd, device=self.device, dtype=torch.float32)
        

    def compute_pd_torque(
        self,
        q_des: torch.Tensor,
        kp: Optional[torch.Tensor] = None,
        kd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute PD control torques manually.
        
        Args:
            q_des: Desired joint positions of shape (n_envs, n_joints)
            kp: Optional position gains (uses stored gains if None)
            kd: Optional velocity gains (uses stored gains if None)
            
        Returns:
            Computed torques of shape (n_envs, n_joints)
        """
        if kp is None:
            kp = self.kp
        if kd is None:
            kd = self.kd
            
        q_pos = self.get_qpos()
        q_vel = self.get_qvel()
        
        # Ensure gains are properly broadcast
        if kp.dim() == 1:
            kp = kp.unsqueeze(0)
        if kd.dim() == 1:
            kd = kd.unsqueeze(0)
        
        torque = kp * (q_des - q_pos) - kd * q_vel
        return torque

    # ---------- Inverse Kinematics ----------

    def inverse_kinematics_batched(
        self,
        target_pos: torch.Tensor,   # (n_envs, 3)
        target_quat: torch.Tensor,  # (n_envs, 4) wxyz
    ) -> torch.Tensor:
        # --- sanity: all must be same n_envs ---
        n = target_pos.shape[0]
        assert target_quat.shape[0] == n
        assert target_pos.shape[1] == 3 and target_quat.shape[1] == 4
        # command = [pos, quat]
        ik_command = torch.cat([target_pos, target_quat], dim=-1)  # (n_envs, 7)

        # set command once
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(ik_command)

        # current state (must be (n_envs, ...))
        ee_pos_w, ee_quat_w = self.get_ee_pose()                   # (n_envs,3), (n_envs,4)
        jacobian = self.get_jacobian()                             # (n_envs, 6, n_joints)
        joint_ids = self.robot_entity_cfg.joint_ids
        joint_pos = self.get_qpos()[:, joint_ids]                  # (n_envs, n_joints)

        # print(f"ee_pos_w: {ee_pos_w.shape}, ee_quat_w: {ee_quat_w.shape}")
        # print(f"jacobian: {jacobian.shape}, joint_pos: {joint_pos.shape}")
        # # EE pose in ROOT frame
        root_pose_w = self.robot.data.root_state_w[:, 0:7]         # (n_envs, 7)
        # print(f"root_pose_w: {root_pose_w.shape}")
        # print(f"root_pose_w: {self.robot.data.root_state_w}")
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pos_w, ee_quat_w
        )

        # one-shot compute
        joint_pos_des = self.diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        return joint_pos_des


    # ---------- Reset Methods ----------
    def reset_all_envs(self):
        """Reset all environments to default state."""
        default_joint_pos = self.robot.data.default_joint_pos.clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        self.robot.reset()
        
        # Set joint position targets to prevent falling after reset
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())

    def reset_envs(self, env_ids: torch.Tensor):
        """
        Reset specific environments to default state.
        
        Args:
            env_ids: Environment indices to reset
        """
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        
        # Set joint position targets to prevent falling after reset
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone(), env_ids=env_ids)

    def quat_angle_error(self, q_actual: np.ndarray, q_target: np.ndarray) -> float:
        """Compute angular error between two quaternions."""
        q_actual = q_actual / (np.linalg.norm(q_actual) + 1e-8)
        q_target = q_target / (np.linalg.norm(q_target) + 1e-8)
        dot_val = np.abs(np.dot(q_actual, q_target))
        return 2 * np.arccos(np.clip(dot_val, 0.0, 1.0))


    def quat_wxyz_from_rotmat(self,R: torch.Tensor) -> torch.Tensor:
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

    def spawn_trajectory_markers_usd(self, sim, tcp_T, every_k=5, radius=0.01, parent_path="/World/Visuals/tcp_traj"):
        """
        Spawn static Sphere prims at tcp_T positions.
        tcp_T: (T,4,4) numpy array in WORLD frame.
        sim: isaaclab.sim.SimulationContext (has sim.stage)
        """
        from pxr import UsdGeom, Gf, Usd
        stage = sim.stage

        # Ensure parent Xform exists
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            parent = UsdGeom.Xform.Define(stage, parent_path).GetPrim()

        for i in range(0, len(tcp_T), every_k):
            p = tcp_T[i, 0:3, 3]
            prim_path = f"{parent_path}/p_{i:04d}"

            # Define a sphere
            sphere = UsdGeom.Sphere.Define(stage, prim_path)
            sphere.GetRadiusAttr().Set(float(radius))

            # Set world position via xform translate op
            xform = UsdGeom.Xformable(sphere.GetPrim())
            # clear any previous ops if re-running
            xform.ClearXformOpOrder()
            t_op = xform.AddTranslateOp()
            t_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))

    # ---------- Utility Methods ----------

    def close(self):
        """Clean up simulation resources."""
        if self.sim is not None:
            self.sim.stop()
