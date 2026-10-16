from dataclasses import dataclass

import numpy as np


@dataclass
class MockWorld:
    width: int = 640
    height: int = 480
    cube_xy: np.ndarray | None = None
    target_xy: tuple[int, int] = (470, 230)
    gripper_xy: np.ndarray | None = None
    attached: bool = False

    def __post_init__(self) -> None:
        self.cube_xy = (
            np.array([180.0, 220.0], dtype=np.float32)
            if self.cube_xy is None
            else np.asarray(self.cube_xy, dtype=np.float32)
        )
        self.gripper_xy = (
            np.array([320.0, 400.0], dtype=np.float32)
            if self.gripper_xy is None
            else np.asarray(self.gripper_xy, dtype=np.float32)
        )

    def render(self) -> np.ndarray:
        from robot_vla.mock.renderer import render_rgb

        return render_rgb(self)

    def update_from_joints(self, joints_deg: np.ndarray) -> None:
        from robot_vla.mock.dynamics import apply_joint_dynamics

        apply_joint_dynamics(self, joints_deg)
