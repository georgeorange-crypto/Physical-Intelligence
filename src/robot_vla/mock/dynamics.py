import numpy as np

from robot_vla.mock.world import MockWorld


def apply_joint_dynamics(world: MockWorld, joints_deg: np.ndarray) -> None:
    x = float(np.interp(joints_deg[0], [0, 180], [40, world.width - 40]))
    y = float(np.interp(joints_deg[1], [0, 180], [world.height - 40, 40]))
    world.gripper_xy = np.array([x, y], dtype=np.float32)
    distance = float(np.linalg.norm(world.gripper_xy - world.cube_xy))
    if joints_deg[5] <= 60 and distance <= 42:
        world.attached = True
    if world.attached:
        world.cube_xy = world.gripper_xy.copy()
    if joints_deg[5] >= 100:
        world.attached = False
