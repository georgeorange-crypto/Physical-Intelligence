from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotAction:
    joints_deg: np.ndarray
    duration_ms: int = 200

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_deg, dtype=np.float32)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("joints_deg must be a finite vector of shape (6,)")
        if not 20 <= self.duration_ms <= 10_000:
            raise ValueError("duration_ms must be between 20 and 10000")
        object.__setattr__(self, "joints_deg", joints)


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[RobotAction, ...]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("action chunk cannot be empty")


@dataclass(frozen=True)
class ActionResult:
    accepted: bool
    command_id: str
    backend: str
    message: str = ""
