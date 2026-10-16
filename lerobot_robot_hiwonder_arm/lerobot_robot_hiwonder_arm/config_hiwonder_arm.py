from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("hiwonder_arm")
@dataclass
class HiwonderArmConfig(RobotConfig):
    mode: str = "mock"
    backend_type: str = "mock"
    base_url: str = "http://127.0.0.1:8765"
    calibration_file: str | None = None
    physical_calibration_verified: bool = False
    image_width: int = 640
    image_height: int = 480
