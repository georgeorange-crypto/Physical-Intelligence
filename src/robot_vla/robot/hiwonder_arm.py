from robot_vla.backend.base import RobotBackend
from robot_vla.backend.mock_backend import MockBackend
from robot_vla.backend.rpc_backend import RpcBackend
from robot_vla.robot.config_hiwonder_arm import HiwonderArmConfig


def make_backend(config: HiwonderArmConfig) -> RobotBackend:
    if config.backend.type == "mock":
        return MockBackend(config.width, config.height)
    if config.backend.type == "rpc":
        return RpcBackend(
            config.backend.base_url,
            calibration_file=config.backend.calibration_file,
            physical_calibration_verified=config.backend.physical_calibration_verified,
            real_mode=config.mode == "real",
        )
    raise ValueError(f"unsupported backend: {config.backend.type}")
