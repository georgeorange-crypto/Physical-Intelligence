import numpy as np
import pytest

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.contracts import RobotAction


def test_backend_contract_and_fault_rejection():
    backend = MockBackend()
    backend.connect()
    assert backend.health().ok
    obs = backend.get_observation()
    assert obs.joints_deg.shape == (6,)
    assert obs.image_rgb.ndim == 3
    assert backend.send_action(RobotAction(np.full(6, 90))).accepted
    backend.faults["action_rejection"] = True
    assert not backend.send_action(RobotAction(np.full(6, 90))).accepted


def test_fault_injection_fails_closed():
    backend = MockBackend()
    backend.connect()
    backend.faults["i2c_error"] = True
    with pytest.raises(OSError):
        backend.get_observation()
    backend.faults.clear()
    backend.faults["none_joint"] = True
    with pytest.raises(RuntimeError, match="missing joint"):
        backend.get_observation()
    backend.faults.clear()
    backend.faults["stale_observation"] = True
    with pytest.raises(TimeoutError):
        from robot_vla.safety.controller import SafetyController

        SafetyController().process(backend.get_observation(), RobotAction(np.full(6, 90)))


def test_fault_injection_disconnect_and_invalid_action():
    backend = MockBackend()
    backend.connect()
    backend.faults["disconnect"] = True
    with pytest.raises(ConnectionError):
        backend.send_action(RobotAction(np.full(6, 90)))
    with pytest.raises(ConnectionError):
        backend.get_observation()

    import time

    from robot_vla.contracts import RobotObservation
    from robot_vla.safety.controller import SafetyController

    observation = RobotObservation(
        time.time_ns(), np.full(6, 90), np.zeros((10, 10, 3), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="envelope"):
        SafetyController().process(observation, RobotAction(np.array([181, 90, 90, 90, 90, 90])))
    with pytest.raises(ValueError):
        RobotAction(np.array([90, 90, 90, 90, 90, np.nan]))


def test_fault_injection_malformed_image_is_rejected():
    backend = MockBackend()
    backend.connect()
    backend.faults["malformed_image"] = True
    with pytest.raises(ValueError, match="image_rgb"):
        backend.get_observation()
