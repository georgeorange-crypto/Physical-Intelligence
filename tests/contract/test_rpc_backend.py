import socket
import threading
import time

import httpx
import pytest
import uvicorn

from robot_vla.backend.rpc_backend import RpcBackend
from robot_vla.contracts import RobotAction
from robot_vla.control.rollout import RolloutController
from robot_vla.policy.mock_policy import MockPolicy
from robot_vla.rpc.mock_armd import create_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server(backend):
    port = _free_port()
    config = uvicorn.Config(create_app(backend), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    return server, thread, f"http://127.0.0.1:{port}"


def test_rpc_and_direct_backends_complete_equivalent_task():
    from robot_vla.backend.mock_backend import MockBackend

    reference = RolloutController(MockBackend(), MockPolicy(), max_steps=100).run("red_to_blue")
    server_backend = MockBackend()
    server, thread, url = _server(server_backend)
    rpc = RpcBackend(url)
    try:
        rpc.connect()
        obs = rpc.get_observation()
        assert obs.joints_deg.shape == (6,)
        assert obs.image_rgb.shape == (480, 640, 3)
        assert rpc.send_action(RobotAction(obs.joints_deg)).accepted
        rpc.disconnect()
        rpc = RpcBackend(url)
        result = RolloutController(rpc, MockPolicy(), max_steps=100).run("red_to_blue")
        assert result.status == reference.status
        assert result.steps == reference.steps
    finally:
        if rpc.connected:
            rpc.disconnect()
        server.should_exit = True
        thread.join(timeout=5)


def test_real_mode_rpc_rejects_actions_without_verified_calibration(tmp_path):
    from robot_vla.backend.mock_backend import MockBackend

    server, thread, url = _server(MockBackend())
    rpc = RpcBackend(url, real_mode=True)
    try:
        rpc.connect()
        obs = rpc.get_observation()
        with pytest.raises(PermissionError, match="not verified"):
            rpc.send_action(RobotAction(obs.joints_deg))
        rpc.disconnect()

        calibration = tmp_path / "calibration.json"
        calibration.write_text("{}", encoding="utf-8")
        rpc = RpcBackend(
            url,
            real_mode=True,
            calibration_file=str(calibration),
            physical_calibration_verified=True,
        )
        rpc.connect()
        obs = rpc.get_observation()
        assert rpc.send_action(RobotAction(obs.joints_deg)).accepted
    finally:
        if rpc.connected:
            rpc.disconnect()
        server.should_exit = True
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("fault", "exception"),
    [("malformed_jpeg", ValueError), ("rpc_timeout_ms", httpx.ReadTimeout)],
)
def test_rpc_fault_injection_fails_closed(fault, exception):
    from robot_vla.backend.mock_backend import MockBackend

    backend = MockBackend()
    backend.faults[fault] = True if fault == "malformed_jpeg" else 1000
    server, thread, url = _server(backend)
    rpc = RpcBackend(url, timeout_s=0.05)
    try:
        rpc.connect()
        with pytest.raises(exception):
            rpc.get_observation()
    finally:
        rpc.disconnect()
        server.should_exit = True
        thread.join(timeout=5)
