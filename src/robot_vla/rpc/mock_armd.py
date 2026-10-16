import base64
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.contracts import RobotAction


class ActionRequest(BaseModel):
    joints_deg: list[float] = Field(min_length=6, max_length=6)
    duration_ms: int = Field(ge=20, le=10_000)


def create_app(backend: MockBackend | None = None) -> FastAPI:
    robot = backend or MockBackend()
    robot.connect()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        robot.disconnect()

    app = FastAPI(title="Mock armd", version="mock-1.0", lifespan=lifespan)
    app.state.backend = robot

    @app.get("/health")
    def health() -> dict[str, object]:
        status = robot.health()
        return {
            "ok": status.ok,
            "backend": status.backend,
            "controller_version": status.controller_version,
        }

    @app.get("/observation")
    def observation() -> dict[str, object]:
        delay_ms = int(robot.faults.get("rpc_timeout_ms", 0))
        if delay_ms:
            time.sleep(delay_ms / 1000)
        obs = robot.get_observation()
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(obs.image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise HTTPException(status_code=500, detail="JPEG encoding failed")
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        if robot.faults.get("malformed_jpeg"):
            image_data = "not-a-jpeg"
        return {
            "joints_deg": obs.joints_deg.tolist(),
            "timestamp_ns": obs.timestamp_ns,
            "image_jpeg": image_data,
        }

    @app.post("/action")
    def action(request: ActionRequest) -> dict[str, object]:
        try:
            result = robot.send_action(RobotAction(request.joints_deg, request.duration_ms))
        except (ValueError, ConnectionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"accepted": result.accepted, "command_id": result.command_id, "backend": "mock"}

    return app


app = create_app()
