from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointLimits:
    minimum_deg: np.ndarray
    maximum_deg: np.ndarray
    source: str = "driver_envelope_only"
    physical_calibration_verified: bool = False

    @classmethod
    def mock_envelope(cls) -> "JointLimits":
        return cls(
            np.zeros(6, dtype=np.float32),
            np.array([180, 180, 180, 180, 270, 180], dtype=np.float32),
        )

    def contains(self, joints_deg: np.ndarray) -> bool:
        return bool(
            np.all(joints_deg >= self.minimum_deg) and np.all(joints_deg <= self.maximum_deg)
        )
