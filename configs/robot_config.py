# Copyright (c) 2024-2025, Yash Jangir
# Robot configuration for Isaac Lab system identification

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RobotConfig:
    """Configuration for a robot in the system identification pipeline."""
    
    dataset_name: str
    """Name of the dataset."""
    
    name: str
    """Name identifier for the robot."""
    
    usd_path: str
    """Path to the robot USD file."""
    
    ee_link: str
    """Name of the end-effector link."""
    
    arm_joint_names: list[str]
    """List of arm joint name patterns."""
    
    gripper_joint_names: list[str]
    """List of gripper joint name patterns."""
    
    n_arm_joints: int
    """Number of arm joints."""
    
    n_gripper_joints: int
    """Number of gripper joints."""
    
    default_kp: Optional[np.ndarray] = None
    """Default position gains for PD control."""
    
    default_kd: Optional[np.ndarray] = None
    """Default velocity gains for PD control."""
    
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Default spawn position."""
    
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Default spawn rotation (wxyz quaternion)."""


# Robot configurations
ROBOT_CONFIGS = {
    "PandaConfig": FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    
    "ur10": RobotConfig(
        name="ur10",
        usd_path="${ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10.usd",
        ee_link="ee_link",
        arm_joint_names=[".*"],
        gripper_joint_names=[],
        n_arm_joints=6,
        n_gripper_joints=0,
        default_kp=np.array([800, 800, 600, 400, 400, 400], dtype=np.float32),
        default_kd=np.array([80, 80, 60, 40, 40, 40], dtype=np.float32),
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    ),
}


def get_robot_config(robot_name: str) -> RobotConfig:
    """
    Get robot configuration by name.
    
    Args:
        robot_name: Name of the robot
        
    Returns:
        RobotConfig for the specified robot
        
    Raises:
        KeyError: If robot_name is not found
    """
    if robot_name not in ROBOT_CONFIGS:
        available = list(ROBOT_CONFIGS.keys())
        raise KeyError(f"Unknown robot '{robot_name}'. Available: {available}")
    return ROBOT_CONFIGS[robot_name]
