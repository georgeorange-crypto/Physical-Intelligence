from typing import Any

import numpy as np
from lerobot.robots import Robot

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.backend.rpc_backend import RpcBackend
from robot_vla.contracts import RobotAction
from robot_vla.safety.controller import SafetyController

from .config_hiwonder_arm import HiwonderArmConfig


class HiwonderArm(Robot):
    config_class = HiwonderArmConfig
    name = "hiwonder_arm"

    def __init__(self, config: HiwonderArmConfig):
        super().__init__(config)
        self.config = config
        if config.backend_type == "mock":
            self.backend = MockBackend(config.image_width, config.image_height)
        elif config.backend_type == "rpc":
            self.backend = RpcBackend(
                config.base_url,
                calibration_file=config.calibration_file,
                physical_calibration_verified=config.physical_calibration_verified,
                real_mode=config.mode == "real",
            )
        else:
            raise ValueError(f"unsupported backend_type: {config.backend_type}")
        self.safety = SafetyController()

    @property
    def observation_features(self) -> dict[str, Any]:
        return {
            **{f"joint_{i}.pos": float for i in range(1, 7)},
            "observation.images.image": (self.config.image_height, self.config.image_width, 3),
        }

    @property
    def action_features(self) -> dict[str, Any]:
        return {f"joint_{i}.pos": float for i in range(1, 7)}

    @property
    def is_connected(self) -> bool:
        return self.backend.connected

    @property
    def is_calibrated(self) -> bool:
        if self.config.mode == "mock":
            return True
        return bool(
            self.config.mode == "real"
            and self.config.physical_calibration_verified
            and self.config.calibration_file
            and __import__("pathlib").Path(self.config.calibration_file).is_file()
        )

    def connect(self, calibrate: bool = True) -> None:
        self.backend.connect()

    def disconnect(self) -> None:
        self.backend.disconnect()

    def calibrate(self) -> None:
        if self.config.mode == "real":
            raise RuntimeError(
                "physical calibration must be performed and verified using the hardware checklist"
            )

    def configure(self) -> None:
        """No runtime configuration is needed for MockBackend or the HTTP contract."""

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("HiwonderArm is not connected")
        obs = self.backend.get_observation()
        return {
            **{f"joint_{i}.pos": float(obs.joints_deg[i - 1]) for i in range(1, 7)},
            "observation.images.image": obs.image_rgb,
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("HiwonderArm is not connected")
        if self.config.mode == "real" and not self.is_calibrated:
            raise PermissionError(
                "real-mode action rejected: verified physical calibration is required"
            )
        obs = self.backend.get_observation()
        raw = RobotAction(
            np.array([action[f"joint_{i}.pos"] for i in range(1, 7)], dtype=np.float32)
        )
        safe = self.safety.process(obs, raw)
        result = self.backend.send_action(safe)
        if not result.accepted:
            raise RuntimeError(f"backend rejected action: {result.message}")
        return {f"joint_{i}.pos": float(safe.joints_deg[i - 1]) for i in range(1, 7)}
