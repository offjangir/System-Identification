#!/usr/bin/env python3
"""
sysid_isaaclab.py – System Identification with Simulated Annealing on DROID trajectories
using Isaac Lab simulator.

This script performs system identification to find optimal PD gains for the Franka Panda
robot that minimize tracking error on DROID trajectory datasets.

Usage:
    ./isaaclab.sh -p scene_generation_isaaclab/sysid_isaaclab.py --num_envs 43 --step_max 25
"""

from __future__ import annotations

import argparse
import os
import math
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

# Import Isaac Lab app launcher first
from isaaclab.app import AppLauncher

# Add argument parser
parser = argparse.ArgumentParser(description="System ID with simulated annealing on DROID trajectories")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")
parser.add_argument("--step_max", type=int, default=1000, help="Number of simulated annealing steps")
parser.add_argument("--data_dir", type=str, default="/data/user_data/yjangir/yash/IsaacLab-Cluster/scene_generation_isaaclab/data/droid_numpy/",
                    help="Path to DROID dataset")
parser.add_argument("--output_dir", type=str, default="./results_isaaclab",
                    help="Directory to save results")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from simulator_isaaclab import SimulatorIsaacLab
from utilities.data_loader import load_droid_numpy_extracted_droid
from utilities.visualizer import spawn_trajectory_markers_usd_tcp, spawn_trajectory_markers_usd_cartesian
from utilities.rotations import eulertotquat, quattoeuler
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
# --------------------------------------------------------------------------- #
#  SysID Multi-Env Class                                                      #
# --------------------------------------------------------------------------- #
class SysIDIsaacLab:
    """
    System identification class using Isaac Lab for parallel trajectory tracking.
    """
    
    def __init__(
        self,
        robot_name: str = "franka",
        n_envs: int = 43,
        show_viewer: bool = False,
        droid_dir: str = "./data/droid_numpy",
        device: str = "cuda:0",
    ):
        """
        Initialize the system identification environment.
        
        Args:
            robot_name: Name of the robot
            n_envs: Number of parallel environments
            show_viewer: Whether to show viewer
            droid_dir: Path to DROID dataset
            device: Compute device
        """
        self.n_envs = n_envs
        self.show_viewer = show_viewer
        self.device = device

        # Create Isaac Lab simulator
        self.sim = SimulatorIsaacLab(
            robot_name=robot_name,
            n_envs=n_envs,
            show_viewer=show_viewer,
            device=device,
            dt=0.01,
        )
        
        self.robot = self.sim.robot
        self.ee_link_name = "panda_hand"
        self.sim.start_sim()

        # Load trajectories
        self.trajectories = load_droid_numpy_extracted_droid(droid_dir)
        trajectory = self.trajectories[0]
        self.cartesian_position = trajectory["cartesian_position"]  # (T,6)
        # print(f"[INFO] Trajectory has {len(self.cartesian_position)} steps") 
        # Track valid environments (some may have IK failures)
        self.valid_mask = np.ones(self.n_envs, dtype=bool)
        


    def _apply_valid_mask(self, arr: np.ndarray) -> np.ndarray:
        """Apply valid mask to array, setting invalid entries to NaN."""
        arr = np.asarray(arr, dtype=np.float64)
        arr[~self.valid_mask] = np.nan
        return arr

    # ------------------------------------------------------------------ #
    #  Environment Operations                                            #
    # ------------------------------------------------------------------ #
    # def reset_all_envs_to_init(self, traj_batch):
    #     """Reset all environments to their initial trajectory poses."""
    #     pos_init, quat_init = [], []
        
    #     # print(f"[INFO] Traj batch: {traj_batch}")

    #     for env_i in range(self.n_envs):
    #         data = traj_batch[env_i]
    #         p = data["cartesian_position"][0][:3]
    #         euler = data["cartesian_position"][0][3:6]
    #         quat = eulertotquat(euler)
    #         pos_init.append(p)
    #         quat_init.append(quat)
    #     # Compute IK for initial poses
    #     # Change dimensions of pos_init and quat_init to (n_envs, 3) and (n_envs, 4)
    #     pos_init = np.asarray(pos_init, dtype=np.float32).reshape(-1, 3)
    #     quat_init = np.asarray(quat_init, dtype=np.float32).reshape(-1, 4)

    #     pos_w = torch.tensor(pos_init, device=self.device, dtype=torch.float32)
    #     quat_w = torch.tensor(quat_init, device=self.device, dtype=torch.float32)

    #     # 1) if traj is env-local, add env origins
    #     pos_w = pos_w + self.sim.scene.env_origins  # (B,3)

    #     # 2) convert world target -> base target
    #     # print Traget ee pose 
    #     root_pose_w = self.sim.robot.data.root_pose_w  # (B,7)
    #     pos_b, quat_b = subtract_frame_transforms(
    #         root_pose_w[:, 0:3], root_pose_w[:, 3:7],
    #         pos_w, quat_w
    #     )

    #     # 3) IK in base frame
    #     q_ik = self.sim.inverse_kinematics_batched(pos_b, quat_b)
        
    #     # Build full joint position tensor (arm + gripper)
    #     q_full = torch.zeros((self.n_envs, 9), device=self.device)
    #     q_full[:, :7] = q_ik
    #     q_full[:, 7:] = 0.04  # Default gripper position
    #     q_vel = torch.zeros_like(q_full)
    #     env_ids = torch.arange(self.n_envs, device=self.device)
    #     self.sim.robot.write_joint_state_to_sim(q_full[:, :7], q_vel[:, :7], joint_ids=self.sim.joint_ids, env_ids=env_ids)
    #     self.sim.step()
    #     self.sim.scene.update(self.sim.dt)
 
    def reset_all_envs_to_init(self, traj_batch):
        """Reset all environments to their initial trajectory poses."""

        joint_pos = self.sim.robot.data.default_joint_pos.clone()
        joint_vel = self.sim.robot.data.default_joint_vel.clone()
        self.sim.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.sim.warmup()

        pos_init, quat_init = [], []
        
        # print(f"[INFO] Traj batch: {traj_batch}")
        for env_i in range(self.n_envs):
            data = traj_batch[env_i]
            p = data["cartesian_position"][0][:3]
            euler = data["cartesian_position"][0][3:6]
            quat = eulertotquat(euler)
            pos_init.append(p)
            quat_init.append(quat)
        # Compute IK for initial poses
        # Change dimensions of pos_init and quat_init to (n_envs, 3) and (n_envs, 4)
        pos_init = np.asarray(pos_init, dtype=np.float32).reshape(-1, 3)
        quat_init = np.asarray(quat_init, dtype=np.float32).reshape(-1, 4)

        pos_w = torch.tensor(pos_init, device=self.device, dtype=torch.float32)
        quat_w = torch.tensor(quat_init, device=self.device, dtype=torch.float32)

        # 1) if traj is env-local, add env origins
        pos_w = pos_w + self.sim.scene.env_origins  # (B,3)

        # 2) convert world target -> base target
        # print Traget ee pose 
        root_pose_w = self.sim.robot.data.root_pose_w  # (B,7)
        pos_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            pos_w, quat_w
        )

        # 3) IK in base frame
        q_ik = self.sim.inverse_kinematics_batched(pos_b, quat_b)
        
        # Build full joint position tensor (arm + gripper)
        q_full = torch.zeros((self.n_envs, 9), device=self.device)
        q_full[:, :7] = q_ik
        q_full[:, 7:] = 0.04  # Default gripper position
        q_vel = torch.zeros_like(q_full)
        env_ids = torch.arange(self.n_envs, device=self.device)
        self.sim.robot.write_joint_state_to_sim(q_full[:, :7], q_vel[:, :7], joint_ids=self.sim.joint_ids, env_ids=env_ids)
        self.sim.step()
        self.sim.scene.update(self.sim.dt)


    def get_traj_batch(self, batch_idx: int):
        start = (batch_idx * self.n_envs) % len(self.trajectories)
        end = start + self.n_envs

        # print(f"[INFO] Batch {batch_idx} - Start: {start}, End: {end}")
        if end <= len(self.trajectories):
            return self.trajectories[start:end]
        else:
            # wrap-around
            return self.trajectories[start:] + self.trajectories[:end - len(self.trajectories)]
    # ------------------------------------------------------------------ #
    #  PD Evaluation                                                     #
    # ------------------------------------------------------------------ #
    def test_pd_params_all(self, kp, kd, batch_idx=0, step_per_wp=3, record_demo=False):
        traj_batch = self.get_traj_batch(batch_idx)   # size == n_envs
        self.sim.diff_ik_controller.reset()
        self.reset_all_envs_to_init(traj_batch)
        self.sim.diff_ik_controller.reset()

        # reset IK controller
        self.sim.warmup()

        kp_t = torch.tensor(kp, device=self.device, dtype=torch.float32).unsqueeze(0).expand(self.n_envs, -1)
        kd_t = torch.tensor(kd, device=self.device, dtype=torch.float32).unsqueeze(0).expand(self.n_envs, -1)
        self.sim.set_pd_all(kp_t, kd_t)
        self.sim.warmup()
        
        max_steps = max(len(d["cartesian_position"]) for d in traj_batch)

        # -------- build padded pose_des (B,T,6) in env-local coords --------
        pose_list = []
        for env_i in range(self.n_envs):
            traj = np.asarray(traj_batch[env_i]["cartesian_position"], dtype=np.float32)  # (Ti,6)
            T = traj.shape[0]
            if T < max_steps:
                pad_len = max_steps - T
                traj = np.concatenate([traj, np.repeat(traj[-1][None, :], pad_len, axis=0)], axis=0)
            pose_list.append(traj)

        pose_des = np.stack(pose_list, axis=0)  # (B,T,6)
        # -------- add env origins ONLY to xyz (env-local -> world) --------
        env_origins = self.sim.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        pose_des[:, :, 0:3] += env_origins[:, None, :]  # broadcast over time

        pose_des = torch.from_numpy(pose_des).to(self.device)  # (B,T,6)

        tot_pos = 0.0
        tot_rot = 0.0

        if record_demo:
            frames = []
            gif_path = os.path.join(f"tcp_track_scene.gif")
            import imageio
            writer = imageio.get_writer(gif_path, mode="I", fps=20)

        for i in range(max_steps):
            if record_demo:
                frame = self.sim.camera.data.output["rgb"][0].detach().cpu().numpy()
                writer.append_data(frame)
                frames.append(frame)
            
            curr_waypoint = pose_des[:, i, :]
            ee_pos_w = curr_waypoint[:, :3]
            ee_quat_w = torch.tensor(eulertotquat(curr_waypoint[:, 3:6].cpu().numpy()), device=self.device, dtype=torch.float32)
            root_pose_w = self.sim.robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pos_w,   ee_quat_w
            )
            q_des = self.sim.inverse_kinematics_batched(ee_pos_b, ee_quat_b)
            q_des_full = torch.zeros((self.n_envs, 9), device=self.device)
            q_des_full[:, :7] = q_des
            q_des_full[:, 7:] = 0.04
            # run subloop
            for _ in range(step_per_wp):
                # use position control
                self.sim.robot.set_joint_position_target(q_des_full[:, :7], joint_ids=self.sim.joint_ids)
                self.sim.step(render=record_demo)
                self.sim.scene.update(self.sim.dt)

                if record_demo == True:
                    self.sim.camera.update(self.sim.dt)
                else:
                    pass

            ee_p, ee_q = self.sim.get_ee_pose()
            ee_p = ee_p.cpu().numpy()
            ee_q = ee_q.cpu().numpy()

            curr_waypoint_np = curr_waypoint.cpu().numpy()
            pos_err = np.linalg.norm(ee_p - curr_waypoint_np[:, :3], axis=1)
            des_q = eulertotquat(curr_waypoint_np[:, 3:6])
            rot_err = np.array([self.quat_angle_error(ee_q[j], des_q[j]) for j in range(self.n_envs)])
            tot_pos += pos_err.mean()
            tot_rot += rot_err.mean()

        if record_demo:
            writer.close()
            video_path = os.path.join(f"tcp_track_scene.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"[INFO] Saved video to {video_path} ({len(frames)} frames)")

        return tot_pos / max_steps, tot_rot / max_steps, (tot_pos + tot_rot) / max_steps


    def quat_angle_error(self, q1, q2):
        """
        Computes angle (rad) between two quaternions.
        q1, q2: (4,) arrays, [x, y, z, w] or [w, x, y, z] consistently.
        """
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        dot = np.abs(np.dot(q1, q2))   # abs handles q and -q equivalence
        dot = np.clip(dot, -1.0, 1.0)  # critical fix

        return 2.0 * np.arccos(dot)
    # ------------------------------------------------------------------ #
    #  Simulated Annealing                                               #
    # ------------------------------------------------------------------ #
    def run_sysid_simulated_annealing(self, step_max: int = 25, batch_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Run system identification using simulated annealing.
        
        Args:
            step_max: Maximum number of annealing steps
            
        Returns:
            Tuple of (best_kp, best_kd) arrays
        """
        full_dof = 9  # 7 arm + 2 gripper
        opt_dof = 7   # Only optimize arm joints

        # Initial PD values for arm
        kp_cur = np.array([800, 800, 600, 600, 300, 300, 300], dtype=int)
        kd_cur = np.array([80, 80, 60, 60, 30, 30, 30], dtype=int)

        # Fixed values for gripper
        kp_fixed = np.array([100, 100])
        kd_fixed = np.array([10, 10])

        # Range for random perturbation
        kp_rng, kd_rng = (200, 1000), (50, 200)

        best_kp = np.concatenate([kp_cur, kp_fixed])
        best_kd = np.concatenate([kd_cur, kd_fixed])
        best_err = float("inf")

        T_init = 1.0
        num_batches = int(np.ceil(len(self.trajectories) / self.n_envs))
        print(f"[INFO] Trajectories: {len(self.trajectories), self.n_envs}")
        print(f"[INFO] Num batches: {num_batches}")

        with tqdm(range(step_max), desc="SimAnn") as pbar:
            for t in pbar:
                batch_idx = np.random.randint(num_batches)

                kp_full = np.concatenate([kp_cur, kp_fixed])
                kd_full = np.concatenate([kd_cur, kd_fixed])
                _, _, comb_e = self.test_pd_params_all(
                    kp_full,
                    kd_full,
                    batch_idx=batch_idx
                )

                pbar.set_postfix(err=f"{comb_e:.4f}", best=f"{best_err:.4f}")

                # --- Propose new candidate (perturb one joint) ---
                kp_new = kp_cur.copy()
                kd_new = kd_cur.copy()
                idx = np.random.randint(opt_dof)  # only first 7 joints
                kp_new[idx] += int((np.random.rand() - 0.5) * 40)
                kd_new[idx] += int((np.random.rand() - 0.5) * 8)
                kp_new = np.clip(kp_new, *kp_rng)
                kd_new = np.clip(kd_new, *kd_rng)

                kp_new_full = np.concatenate([kp_new, kp_fixed])
                kd_new_full = np.concatenate([kd_new, kd_fixed])
                # --- Evaluate candidate ---
                _, _, comb_new = self.test_pd_params_all(
                    kp_new_full,
                    kd_new_full,
                    batch_idx=batch_idx
                )

                # --- Acceptance criterion ---
                delta = comb_e - comb_new
                temp  = T_init * (1 - t / step_max)
                print(f"[INFO] Temp: {temp}")
                print(f"[INFO] Kp new: {kp_new}")
                print(f"[INFO] Kd new: {kd_new}")
                if delta > 0 or np.random.rand() < np.exp(delta / (temp + 1e-9)):
                    kp_cur = kp_new
                    kd_cur = kd_new
                
                if comb_e < best_err:
                    best_kp = kp_full.copy()
                    best_kd = kd_full.copy()
                    best_err = comb_e

                pbar.set_postfix(err=f"{comb_e:.4f}", best=f"{best_err:.4f}")
                # Evaluate and plot
                if t % 1 == 0:
                    print(f"[INFO] Step {t} of {step_max} - Evaluating and plotting")
                    print(f"[INFO] Best Kp: {best_kp}")
                    print(f"[INFO] Best Kd: {best_kd}")
                    print(f"[INFO] Current error: {comb_e:.4f}, Best error: {best_err:.4f}")
                    print(f"[INFO] Evaluating and plotting")

                    self.evaluate_and_plot(best_kp, best_kd, step_per_wp=3, out_dir="./results_isaaclab", run_id=t)
                    print(f"[INFO] Step {t} of {step_max} - Current error: {comb_e:.4f}, Best error: {best_err:.4f}")

        print(f"\n[RESULT] Best combined error = {best_err:.4f}")
        print(f"[RESULT] Disabled envs: {np.where(~self.valid_mask)[0].tolist()}")
        print(f"[RESULT] Best Kp: {best_kp}")
        print(f"[RESULT] Best Kd: {best_kd}")
        
        return best_kp, best_kd

    # ------------------------------------------------------------------ #
    #  Final Evaluation + Plotting                                       #
    # ------------------------------------------------------------------ #

    def evaluate_and_plot(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        step_per_wp: int = 3,
        out_dir: str = "./results_isaaclab",
        run_id: int = 0,
    ):
        """
        Evaluate the best PD parameters and generate plots.
        
        Args:
            kp: Position gains
            kd: Velocity gains
            step_per_wp: Simulation steps per waypoint
            out_dir: Output directory for plots
            run_id: Run identifier for output filenames
        """
        os.makedirs(out_dir, exist_ok=True)
        #  take a random batch index
        batch_idx = np.random.randint(0, 10)
        traj_batch = self.get_traj_batch(batch_idx) 

        self.sim.diff_ik_controller.reset()
        self.reset_all_envs_to_init(traj_batch)
        self.sim.diff_ik_controller.reset()
        
        kp_t = torch.tensor(kp, device=self.device, dtype=torch.float32).unsqueeze(0)
        kd_t = torch.tensor(kd, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.sim.set_pd_all(kp_t, kd_t)
        self.sim.warmup()
        max_steps = max(len(d["cartesian_position"]) for d in traj_batch)

        pose_list = []
        for env_i in range(self.n_envs):
            traj = np.asarray(traj_batch[env_i]["cartesian_position"], dtype=np.float32)  # (Ti,6)
            T = traj.shape[0]
            if T < max_steps:
                pad_len = max_steps - T
                traj = np.concatenate([traj, np.repeat(traj[-1][None, :], pad_len, axis=0)], axis=0)
            pose_list.append(traj)
        
        pose_des = np.stack(pose_list, axis=0)  # (B,T,6)
        # -------- add env origins ONLY to xyz (env-local -> world) --------
        env_origins = self.sim.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        pose_des[:, :, 0:3] += env_origins[:, None, :]  # broadcast over time

        pose_des = torch.from_numpy(pose_des).to(self.device)  # (B,T,6)

        pos_errors = []
        rot_errors = []
        ee_pos_list = []
        
        gif_path = os.path.join(out_dir, f"tcp_track_scene_{run_id}.gif")
        writer = imageio.get_writer(gif_path, mode="I", fps=20)
        print(f"[INFO] Writing gif to {gif_path}")
        
        for i in range(max_steps):
            curr_waypoint = pose_des[:, i, :]
            ee_pos_w = curr_waypoint[:, :3]
            ee_quat_w = torch.tensor(eulertotquat(curr_waypoint[:, 3:6].cpu().numpy()), device=self.device, dtype=torch.float32)
            root_pose_w = self.sim.robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pos_w, ee_quat_w
            )
            q_des = self.sim.inverse_kinematics_batched(ee_pos_b, ee_quat_b)
            q_des_full = torch.zeros((self.n_envs, 9), device=self.device)
            q_des_full[:, :7] = q_des
            q_des_full[:, 7:] = 0.04
            
            # Run subloop
            for _ in range(step_per_wp):
                self.sim.robot.set_joint_position_target(q_des_full[:, :7], joint_ids=self.sim.joint_ids)
                self.sim.step(render=True)
                self.sim.scene.update(self.sim.dt)

            frame = self.sim.camera.data.output["rgb"][0].detach().cpu().numpy()
            writer.append_data(frame)

            ee_p, ee_q = self.sim.get_ee_pose()
            ee_p = ee_p.cpu().numpy()
            ee_q = ee_q.cpu().numpy()
            curr_waypoint_np = curr_waypoint.cpu().numpy()

            pos_err = np.linalg.norm(ee_p - curr_waypoint_np[:, :3], axis=1)
            des_q = eulertotquat(curr_waypoint_np[:, 3:6])
            rot_err = np.array([self.quat_angle_error(ee_q[j], des_q[j]) for j in range(self.n_envs)])
            
            pos_errors.append(pos_err.mean())
            rot_errors.append(rot_err.mean())
            ee_pos_list.append(ee_p)

        writer.close()
        
        pos_errors = np.array(pos_errors)
        rot_errors = np.array(rot_errors)
        ee_pos_list = np.array(ee_pos_list)  # (max_steps, n_envs, 3)

        # Transpose ee_pos_list to (n_envs, max_steps, 3)
        ee_pos_list = ee_pos_list.transpose(1, 0, 2)
        pose_des_np = pose_des[:, :, :3].detach().cpu().numpy()  # (n_envs, max_steps, 3)

        # Extract first environment for plotting
        ee_pos_env0 = ee_pos_list[0, :, :]
        pose_des_env0 = pose_des_np[0, :, :]
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.plot(ee_pos_env0[:, 0], label="X_Actual", linestyle="-")
        plt.plot(ee_pos_env0[:, 1], label="Y_Actual", linestyle="-")
        plt.plot(ee_pos_env0[:, 2], label="Z_Actual", linestyle="-")
        plt.plot(pose_des_env0[:, 0], label="X_Desired", linestyle="--")
        plt.plot(pose_des_env0[:, 1], label="Y_Desired", linestyle="--")
        plt.plot(pose_des_env0[:, 2], label="Z_Desired", linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Position (m)")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.title("End Effector Position: Actual vs Desired")
        plt.savefig(os.path.join(out_dir, f"ee_pos_and_waypoints_{run_id}.png"))
        plt.close()
        
        # Print summary statistics
        print(f"[INFO] Mean position error: {pos_errors.mean():.4f} m")
        print(f"[INFO] Mean rotation error: {rot_errors.mean():.4f} rad")


    def record_demo(self):
        """
        Record the demo.
        """
        import imageio
        gif_path = os.path.join(f"tcp_track_scene.gif")
        writer = imageio.get_writer(gif_path, mode="I", fps=20)
        # self.reset_all_envs_to_init()
        # reset_here manually
        # use self.cartesian_position
        # draw spheres
        frames = []
        spawn_trajectory_markers_usd_cartesian(self.sim.sim, self.cartesian_position, radius=0.01)

        self.sim.step(render=True)
        self.sim.scene.update(self.sim.dt)
        self.sim.camera.update(self.sim.dt)
        frame = self.sim.camera.data.output["rgb"][0].detach().cpu().numpy()
        frames.append(frame)
        writer.append_data(frame)
        # save
        imageio.mimsave(gif_path, frames, fps=20)

        frames = []

        for i in range(len(self.cartesian_position)):
            curr_waypoint = self.cartesian_position[i]
            ee_pos_w = torch.tensor(curr_waypoint[:3], dtype=torch.float32).to(self.device)
            ee_quat_w = torch.tensor(eulertotquat(curr_waypoint[3:6]), dtype=torch.float32).to(self.device)
            target_pos_w = ee_pos_w.view(1, 3).repeat(self.n_envs, 1)
            target_quat_w = ee_quat_w.view(1, 4).repeat(self.n_envs, 1)
            root_pose_w = self.sim.robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                target_pos_w.repeat(self.n_envs, 1), target_quat_w.repeat(self.n_envs, 1)
            )
            jacobian = self.sim.robot.root_physx_view.get_jacobians()[:, self.sim.ee_jacobi_idx, :, self.sim.joint_ids]
            joint_pos = self.sim.robot.data.joint_pos[:, self.sim.joint_ids]
            des_q = self.sim.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            vel_target = torch.zeros_like(des_q)
            vel_target[:, 7:] = 0.04
            # self.sim.robot.write_joint_state_to_sim(des_q[:, :7], vel_target[:, :7], joint_ids=self.sim.joint_ids)
            self.sim.robot.set_joint_position_target(des_q[:, :7], joint_ids=self.sim.joint_ids)
            self.sim.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.sim.scene.update(self.sim.dt)
            self.sim.camera.update(self.sim.dt)
            frame = self.sim.camera.data.output["rgb"][0].detach().cpu().numpy()
            writer.append_data(frame)
            frames.append(frame)
        writer.close()

        video_path = os.path.join(f"tcp_track_scene.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"[INFO] Saved video to {video_path} ({len(frames)} frames)")
    
    def record_demo_position_control(self):
        """
        Record the demo for position control.
        """
        import imageio
        gif_path = os.path.join(f"tcp_track_scene_position_control.gif")
        writer = imageio.get_writer(gif_path, mode="I", fps=20)
        traj_batch = self.get_traj_batch(0)
        self.reset_all_envs_to_init(traj_batch)

        frames = []
        # get pos and quat for all envs
        max_steps = max(len(data["cartesian_position"]) for data in traj_batch)
        pose_list = []

        for env_i in range(self.n_envs):
            traj = np.array(traj_batch[env_i]["cartesian_position"])  # (Ti, 6)
            T = traj.shape[0]

            if T < max_steps:
                pad_len = max_steps - T
                last = traj[-1][None, :]                     # (1,6)
                pad = np.repeat(last, pad_len, axis=0)       # (pad_len,6)
                traj = np.concatenate([traj, pad], axis=0)   # (max_steps,6)

            pose_list.append(traj)

        pose_des = np.stack(pose_list, axis=0)  # (B, max_steps, 6)

        env_origins = self.sim.scene.env_origins.detach().cpu().numpy()  # (B,3)

        env_pad = np.zeros((self.n_envs, 1, 6), dtype=pose_des.dtype)
        env_pad[:, 0, :3] = env_origins

        pose_des = pose_des + env_pad
        pose_des = torch.tensor(pose_des, device=self.device, dtype=torch.float32)

        pose_des = torch.tensor(pose_des, device=self.device, dtype=torch.float32).reshape(self.n_envs, max_steps, 6)
        print("[INFO] pose_des:", pose_des.shape)

        for i in range(max_steps):
            curr_waypoint = pose_des[:, i, :]
            ee_pos_w = curr_waypoint[:, :3]
            ee_quat_w = torch.tensor(eulertotquat(curr_waypoint[:, 3:6].cpu().numpy()), device=self.device, dtype=torch.float32)
            root_pose_w = self.sim.robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pos_w,   ee_quat_w
            )
            q_des = self.sim.inverse_kinematics_batched(ee_pos_b, ee_quat_b)
            q_des_full = torch.zeros((self.n_envs, 9), device=self.device)
            q_des_full[:, :7] = q_des
            q_des_full[:, 7:] = 0.04
            # use position control
            self.sim.robot.set_joint_position_target(q_des_full[:, :7], joint_ids=self.sim.joint_ids)
            self.sim.step(render=True)
            self.sim.scene.update(self.sim.dt)
            self.sim.camera.update(self.sim.dt)
            frame = self.sim.camera.data.output["rgb"][0].detach().cpu().numpy()
            writer.append_data(frame)
            frames.append(frame)
        writer.close()
        video_path = os.path.join(f"tcp_track_scene_position_control.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"[INFO] Saved video to {video_path} ({len(frames)} frames)")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    """Main function."""
    print("[INFO] Starting Isaac Lab System Identification...")
    
    # Count available trajectories
    try:
        n_traj = len(os.listdir(args_cli.data_dir))
        print("[INFO] data_dir:", args_cli.data_dir)
        print("[INFO] n_traj:", n_traj//3)
        print(f"[INFO] Found {n_traj//3} trajectories in dataset")
    except Exception as e:
        print(f"[WARN] Could not count trajectories: {e}")
        n_traj = args_cli.num_envs
    
    # Use minimum of available trajectories and requested envs
    n_envs = min(n_traj, args_cli.num_envs)
    
    # Create SysID environment
    env = SysIDIsaacLab(
        n_envs=n_envs,
        droid_dir=args_cli.data_dir,
        show_viewer=not args_cli.headless,
    )

    # env.record_demo_position_control()
    # env.record_demo_position_control()
    # env.record_demo_position_control_arm_1_only()
    # env.record_demo_force_control(kp=[900, 900, 700, 700, 400, 400, 400, 100, 100], kd=[90, 90, 70, 70, 40, 40, 40, 10, 10], step_per_wp=20)
    # error = env.test_pd_params_all(kp=[900, 900, 700, 700, 400, 400, 400, 100, 100], kd=[90, 90, 70, 70, 40, 40, 40, 10, 10], batch_idx=0, step_per_wp=2, record_demo=True)
    # print("[INFO] Error:", error)

    # Run simulated annealing
    best_kp, best_kd = env.run_sysid_simulated_annealing(step_max=args_cli.step_max)
    # save best kp and kd in text file
    with open(os.path.join(args_cli.output_dir, "best_kp.txt"), "w") as f:
        f.write(f"Best Kp: {best_kp}\n")
    with open(os.path.join(args_cli.output_dir, "best_kd.txt"), "w") as f:
        f.write(f"Best Kd: {best_kd}\n")
    # env.evaluate_and_plot(kp=[900, 900, 700, 700, 400, 400, 400, 100, 100], kd=[90, 90, 70, 70, 40, 40, 40, 10, 10], step_per_wp=3, out_dir="./results_isaaclab", t=1)
    
    # Evaluate and plot
    # env.evaluate_and_plot(best_kp, best_kd, steps=30, step_per_wp=20, out_dir=args_cli.output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()

