# Isaac Lab System Identification & Trajectory Simulation

This directory contains the Isaac Lab port of the Genesis-based system identification pipeline. It provides tools for system identification and trajectory replay on DROID datasets using NVIDIA Isaac Lab.

## Overview

The codebase consists of three main components:

1. **`simulator_isaaclab.py`**: Multi-environment simulator wrapper for Isaac Lab
2. **`sysid_isaaclab.py`**: System identification using simulated annealing
3. **`replay_isaaclab.py`**: Single trajectory replay and evaluation

## Prerequisites

### Isaac Lab Installation
Ensure Isaac Lab is installed and configured properly. Follow the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) for installation instructions.

### Python Dependencies
```bash
pip install tensorflow tensorflow-datasets transforms3d matplotlib tqdm imageio
```

### DROID Dataset
The scripts expect the DROID dataset in RLDS format. Update the `--data_dir` argument to point to your dataset location.

## Usage

### System Identification

Run system identification with simulated annealing to find optimal PD gains:

```bash
# Navigate to Isaac Lab directory
cd /path/to/IsaacLab

# Run system identification (headless mode)
./isaaclab.sh -p /path/to/scene_generation_isaaclab/sysid_isaaclab.py \
    --num_envs 43 \
    --step_max 25 \
    --data_dir /path/to/droid_100/1.0.0 \
    --output_dir ./results_isaaclab \
    --headless

# Run with viewer (slower but visual)
./isaaclab.sh -p /path/to/scene_generation_isaaclab/sysid_isaaclab.py \
    --num_envs 10 \
    --step_max 10
```

**Arguments:**
- `--num_envs`: Number of parallel environments (default: 43)
- `--step_max`: Number of simulated annealing iterations (default: 25)
- `--data_dir`: Path to DROID RLDS dataset
- `--output_dir`: Directory to save results
- `--headless`: Run without GUI

### Trajectory Replay

Replay a single trajectory and generate error metrics:

```bash
# With good PD parameters (from system identification)
./isaaclab.sh -p /path/to/scene_generation_isaaclab/replay_isaaclab.py \
    --traj_num 3 \
    --out_dir ./results_isaaclab \
    --headless

# With worse (default) PD parameters
./isaaclab.sh -p /path/to/scene_generation_isaaclab/replay_isaaclab.py \
    --traj_num 3 \
    --bad_pd \
    --out_dir ./results_isaaclab \
    --headless
```

**Arguments:**
- `--traj_num`: Trajectory index to replay (default: 2)
- `--bad_pd`: Use worse PD parameters for comparison
- `--out_dir`: Output directory for plots and videos
- `--data_dir`: Path to DROID dataset
- `--save_video`: Save video of trajectory replay

## Output Files

### System Identification
- `error_metrics_isaaclab.png`: Plot of tracking errors over time
- `best_sys_parameters_isaaclab.txt`: Optimal Kp and Kd values

### Trajectory Replay
- `error_metrics_traj{N}.png`: Error metrics plot for trajectory N
- `trajectory_traj{N}.png`: X-Y trajectory comparison plot
- `traj{N}.mp4`: Video of trajectory replay (if `--save_video`)

## Key Differences from Genesis Version

| Feature | Genesis | Isaac Lab |
|---------|---------|-----------|
| Backend | Genesis simulator | NVIDIA Isaac Sim |
| IK Solver | Built-in `robot.inverse_kinematics()` | Differential IK controller |
| Multi-env | Custom batching | Native `InteractiveScene` |
| PD Control | `robot.set_dofs_kp/kv()` | Manual torque computation |
| Rendering | `gs.tools.animate()` | Camera sensor + imageio |

## API Reference

### SimulatorIsaacLab

```python
from simulator_isaaclab import SimulatorIsaacLab

# Create simulator
sim = SimulatorIsaacLab(
    robot_name="franka",
    n_envs=10,
    device="cuda:0",
    show_viewer=False,
    dt=0.01,
)

# Initialize
sim.start_sim()

# Get state
qpos = sim.get_qpos()           # (n_envs, n_joints)
qvel = sim.get_qvel()           # (n_envs, n_joints)
ee_pos, ee_quat = sim.get_ee_pose()  # (n_envs, 3), (n_envs, 4)

# Control
sim.set_qpos(qpos)              # Direct position setting
sim.control_position(qpos_target)  # PD position control
sim.control_force(torques)      # Torque control

# Inverse kinematics
q_des = sim.inverse_kinematics_batched(target_pos, target_quat)

# Step simulation
sim.step(steps=10)
```

## Troubleshooting

### Out of Memory
Reduce `--num_envs` if running out of GPU memory.

### IK Failures
The scripts automatically mask environments with IK failures. Check the output for "Invalid IK" warnings.

### TensorFlow Warnings
TensorFlow may print warnings about GPU memory allocation. These can usually be ignored.

## Citation

If you use this code, please cite:

```bibtex
@misc{isaaclab_sysid,
  title={Isaac Lab System Identification for DROID Trajectories},
  author={Yash Jangir},
  year={2025},
  url={https://github.com/offjangir/robotarena}
}
```
# System-Identification
