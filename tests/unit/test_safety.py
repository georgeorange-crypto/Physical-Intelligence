import numpy as np
import pytest

from robot_vla.contracts import RobotAction, RobotObservation
from robot_vla.safety.controller import SafetyController


def obs(timestamp: int | None = None) -> RobotObservation:
    import time

    return RobotObservation(
        timestamp or time.time_ns(), np.full(6, 90), np.zeros((10, 10, 3), np.uint8)
    )


def test_safety_clips_delta_and_rejects_envelope_violation():
    safety = SafetyController(max_delta_deg=5)
    safe = safety.process(obs(), RobotAction(np.full(6, 120)))
    assert np.allclose(safe.joints_deg, 95)
    with pytest.raises(ValueError, match="envelope"):
        safety.process(obs(), RobotAction(np.array([190, 90, 90, 90, 90, 90])))


def test_safety_rejects_stale_and_estopped():
    safety = SafetyController()
    with pytest.raises(TimeoutError):
        safety.process(obs(1), RobotAction(np.full(6, 90)))
    safety.emergency_stop()
    with pytest.raises(RuntimeError, match="emergency stop"):
        safety.process(obs(), RobotAction(np.full(6, 90)))
