import base64
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np

from robot_vla.contracts import ActionResult, HealthStatus, RobotAction, RobotObservation


class RpcBackend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        calibration_file: str | None = None,
        physical_calibration_verified: bool = False,
        real_mode: bool = False,
        timeout_s: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.calibration_file = Path(calibration_file) if calibration_file else None
        self.physical_calibration_verified = physical_calibration_verified
        self.real_mode = real_mode
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout_s)
        self.connected = False

    def connect(self) -> None:
        status = self.health()
        if not status.ok:
            raise ConnectionError("RPC controller reports unhealthy")
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.client.close()

    def health(self) -> HealthStatus:
        try:
            data = self.client.get("/health").raise_for_status().json()
            return HealthStatus(
                bool(data["ok"]), str(data["backend"]), str(data["controller_version"])
            )
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return HealthStatus(False, "rpc", "unknown", str(exc))

    def get_observation(self) -> RobotObservation:
        if not self.connected:
            raise ConnectionError("RpcBackend is not connected")
        data = self.client.get("/observation").raise_for_status().json()
        try:
            raw = base64.b64decode(data["image_jpeg"], validate=True)
            image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("malformed JPEG")
            return RobotObservation(
                int(data["timestamp_ns"]),
                np.asarray(data["joints_deg"], dtype=np.float32),
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid observation response: {exc}") from exc

    def send_action(self, action: RobotAction) -> ActionResult:
        if not self.connected:
            raise ConnectionError("RpcBackend is not connected")
        if self.real_mode and not self.physical_calibration_verified:
            raise PermissionError("real-mode action rejected: physical calibration is not verified")
        if self.real_mode and (
            self.calibration_file is None or not self.calibration_file.is_file()
        ):
            raise PermissionError("real-mode action rejected: calibration file is required")
        response = self.client.post(
            "/action",
            json={"joints_deg": action.joints_deg.tolist(), "duration_ms": action.duration_ms},
        )
        response.raise_for_status()
        data = response.json()
        return ActionResult(
            bool(data["accepted"]), str(data.get("command_id", uuid.uuid4())), str(data["backend"])
        )
