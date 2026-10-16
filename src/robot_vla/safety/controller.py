import time

import numpy as np

from robot_vla.contracts import RobotAction, RobotObservation
from robot_vla.safety.joint_limits import JointLimits
from robot_vla.safety.rate_limiter import RateLimiter
from robot_vla.safety.watchdog import Watchdog


class SafetyController:
    """Validates and limits every policy command before backend dispatch."""

    def __init__(self, max_delta_deg: float = 5.0, observation_timeout_ms: int = 500) -> None:
        self.max_delta_deg = float(max_delta_deg)
        self.observation_timeout_ms = int(observation_timeout_ms)
        self.estopped = False
        self.last_command_ns = time.time_ns()
        self.limits = JointLimits.mock_envelope()
        self.rate_limiter = RateLimiter(max_delta_deg)
        self.watchdog = Watchdog(timeout_s=max(0.5, observation_timeout_ms / 1000))

    def emergency_stop(self) -> None:
        self.estopped = True

    def reset_estop(self) -> None:
        self.estopped = False

    def process(self, observation: RobotObservation, action: RobotAction) -> RobotAction:
        if self.estopped:
            raise RuntimeError("emergency stop is active")
        if self.watchdog.expired:
            raise TimeoutError("control watchdog expired")
        age_ms = (time.time_ns() - observation.timestamp_ns) / 1e6
        if age_ms < 0 or age_ms > self.observation_timeout_ms:
            raise TimeoutError(f"observation is stale ({age_ms:.1f} ms)")
        target = np.asarray(action.joints_deg, dtype=np.float32)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("action must contain six finite joint targets")
        if not self.limits.contains(target):
            raise ValueError("action exceeds configured driver envelope")
        if not 20 <= action.duration_ms <= 10_000:
            raise ValueError("duration outside allowed range")
        safe = RobotAction(
            self.rate_limiter.apply(observation.joints_deg, target), action.duration_ms
        )
        self.last_command_ns = time.time_ns()
        self.watchdog.heartbeat()
        return safe
