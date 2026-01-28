import numpy as np
import torch
from pxr import UsdGeom, Gf

def spawn_trajectory_markers_usd_tcp(sim, tcp_T, every_k=5, radius=0.01, parent_path="/World/Visuals/tcp_traj"):
    """
    Spawn static Sphere prims at tcp_T positions.
    tcp_T: (T,4,4) numpy array in WORLD frame.
    sim: isaaclab.sim.SimulationContext (has sim.stage)
    """
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


def spawn_trajectory_markers_usd_cartesian(sim, cartesian_position, every_k=5, radius=0.01, parent_path="/World/Visuals/tcp_traj"):
    """
    Spawn static Sphere prims at cartesian_position positions.
    cartesian_position: (T,7) numpy array in WORLD frame.
    sim: isaaclab.sim.SimulationContext (has sim.stage)
    """
    stage = sim.stage

    # Ensure parent Xform exists
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        parent = UsdGeom.Xform.Define(stage, parent_path).GetPrim()

    
    for i in range(0, len(cartesian_position), every_k):
        p = cartesian_position[i, :3]
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


