import time
import uuid

import numpy as np

from robot_vla.backend.base import RobotBackend
from robot_vla.contracts import ActionResult, HealthStatus, RobotAction, RobotObservation
from robot_vla.mock import MockWorld


class MockBackend(RobotBackend):
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.world = MockWorld(width=width, height=height)
        self.joints_deg = np.array([90, 90, 90, 90, 90, 120], dtype=np.float32)
        self.connected = False
        self.faults: dict[str, object] = {}

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def health(self) -> HealthStatus:
        return HealthStatus(self.connected, "mock", "mock-1.0")

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        if self.faults.get("i2c_error"):
            raise OSError("injected I2C-like feedback failure")
        if self.faults.get("none_joint"):
            raise RuntimeError("injected missing joint feedback")
        image = self.world.render()
        if self.faults.get("malformed_image"):
            image = image[:, :, :2]
        ts = 1 if self.faults.get("stale_observation") else time.time_ns()
        return RobotObservation(ts, self.joints_deg.copy(), image)

    def send_action(self, action: RobotAction) -> ActionResult:
        self._require_connected()
        if self.faults.get("i2c_error"):
            raise OSError("injected I2C-like command failure")
        if self.faults.get("disconnect"):
            self.connected = False
            raise ConnectionError("injected disconnect")
        if self.faults.get("action_rejection"):
            return ActionResult(False, str(uuid.uuid4()), "mock", "injected rejection")
        target = np.asarray(action.joints_deg, dtype=np.float32)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("invalid action")
        self.joints_deg = target.copy()
        self.world.update_from_joints(self.joints_deg)
        return ActionResult(True, str(uuid.uuid4()), "mock")

    def _require_connected(self) -> None:
        if not self.connected:
            raise ConnectionError("MockBackend is not connected")
