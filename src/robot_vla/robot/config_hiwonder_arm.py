from dataclasses import dataclass


@dataclass(frozen=True)
class BackendConfig:
    type: str = "mock"
    base_url: str = "http://127.0.0.1:8765"
    calibration_file: str | None = None
    physical_calibration_verified: bool = False


@dataclass(frozen=True)
class HiwonderArmConfig:
    mode: str = "mock"
    backend: BackendConfig = BackendConfig()
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        if self.mode not in {"mock", "real"}:
            raise ValueError("mode must be 'mock' or 'real'")
        if self.backend.type not in {"mock", "rpc"}:
            raise ValueError("backend type must be 'mock' or 'rpc'")
