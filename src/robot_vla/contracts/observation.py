from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotObservation:
    timestamp_ns: int
    joints_deg: np.ndarray
    image_rgb: np.ndarray

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_deg, dtype=np.float32)
        image = np.asarray(self.image_rgb)
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("joints_deg must be a finite float vector of shape (6,)")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("image_rgb must be HxWx3 uint8")
        object.__setattr__(self, "joints_deg", joints)
        object.__setattr__(self, "image_rgb", image)
