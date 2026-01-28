# Requirements to Run Isaac Lab System Identification Code

## 1. Isaac Lab Installation

**Critical**: You must have Isaac Lab installed and properly configured. This is the core dependency.

- Follow the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) for installation
- Ensure Isaac Lab is set up in your environment
- The code requires access to Isaac Lab's simulation framework and assets

## 2. Python Dependencies

Install the following Python packages:

```bash
pip install tensorflow tensorflow-datasets transforms3d matplotlib tqdm imageio
```

### Required Packages:
- **tensorflow**: For TensorFlow Datasets (RLDS format)
- **tensorflow-datasets**: For loading DROID dataset in RLDS format
- **transforms3d**: For quaternion and Euler angle conversions
- **matplotlib**: For plotting error metrics and trajectories
- **tqdm**: For progress bars during simulated annealing
- **imageio**: For saving video files of trajectory replays

### Additional Implicit Dependencies:
- **torch** (PyTorch): Required by Isaac Lab
- **numpy**: Required by Isaac Lab and the code
- **isaaclab**: The Isaac Lab framework itself

## 3. DROID Dataset

You need access to the DROID dataset in RLDS (Reinforcement Learning Datasets) format.

- Default path in code: `/mnt/tank/khtu/robot-arena/droid_100/1.0.0`
- Update the `--data_dir` argument to point to your dataset location
- The dataset should contain trajectories with `cartesian_position` observations

## 4. Hardware Requirements

- **GPU**: CUDA-capable GPU (recommended for performance)
  - The code uses `cuda:0` by default
  - Multiple parallel environments require significant GPU memory
- **Memory**: Sufficient RAM and GPU memory for parallel environments
  - Default uses 43 parallel environments
  - Reduce `--num_envs` if running out of memory

## 5. Isaac Lab Assets

The code requires access to Isaac Lab's asset library:
- Franka Panda robot USD file: `${ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd`
- This should be available if Isaac Lab is properly installed

## 6. Running the Code

### System Identification:
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
```

### Trajectory Replay:
```bash
./isaaclab.sh -p /path/to/scene_generation_isaaclab/replay_isaaclab.py \
    --traj_num 3 \
    --out_dir ./results_isaaclab \
    --headless
```

## 7. Environment Setup Checklist

- [ ] Isaac Lab installed and configured
- [ ] Python dependencies installed (`tensorflow`, `tensorflow-datasets`, `transforms3d`, `matplotlib`, `tqdm`, `imageio`)
- [ ] DROID dataset available and path configured
- [ ] CUDA GPU available (if using GPU acceleration)
- [ ] Isaac Lab assets accessible (Franka robot USD files)
- [ ] Sufficient disk space for output files (plots, videos, parameter files)

## 8. Common Issues

### Out of Memory
- Reduce `--num_envs` parameter (try 10-20 instead of 43)

### IK Failures
- The code automatically masks environments with IK failures
- Check output for "Invalid IK" warnings

### TensorFlow Warnings
- TensorFlow may print warnings about GPU memory allocation
- These can usually be ignored

### Missing Dataset
- Ensure the DROID dataset path is correct
- The dataset must be in RLDS format

## 9. Output Files

After running, you'll get:
- **System Identification**: `error_metrics_isaaclab.png`, `best_sys_parameters_isaaclab.txt`
- **Trajectory Replay**: `error_metrics_traj{N}.png`, `trajectory_traj{N}.png`, `traj{N}.mp4` (if `--save_video`)
