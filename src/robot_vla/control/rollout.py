import time
from dataclasses import dataclass
from pathlib import Path

from robot_vla.backend.base import RobotBackend
from robot_vla.contracts import TaskStatus
from robot_vla.logging.trajectory_logger import TrajectoryLogger
from robot_vla.policy.base import PolicyAdapter
from robot_vla.safety.controller import SafetyController
from robot_vla.task.manager import TaskManager
from robot_vla.task.success_detector import RedToBlueSuccessDetector


@dataclass(frozen=True)
class RolloutResult:
    status: TaskStatus
    steps: int
    backend: str


class RolloutController:
    def __init__(
        self,
        backend: RobotBackend,
        policy: PolicyAdapter,
        safety: SafetyController | None = None,
        detector: RedToBlueSuccessDetector | None = None,
        logger: TrajectoryLogger | None = None,
        max_steps: int = 120,
        timeout_s: float = 60.0,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.safety = safety or SafetyController()
        self.detector = detector or RedToBlueSuccessDetector()
        self.logger = logger
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.tasks = TaskManager()

    def run(self, command: str) -> RolloutResult:
        from robot_vla.task.registry import TaskRegistry

        task = self.tasks.start(TaskRegistry().resolve(command))
        self.policy.reset(task)
        self.backend.connect()
        steps = 0
        started = time.monotonic()
        try:
            obs = self.backend.get_observation()
            if self.detector.is_success(obs):
                self.tasks.status = TaskStatus.SUCCESS
            while self.tasks.status == TaskStatus.RUNNING and steps < self.max_steps:
                if time.monotonic() - started > self.timeout_s:
                    self.tasks.status = TaskStatus.TIMEOUT
                    break
                raw = self.policy.predict(obs, task).actions[0]
                safe = self.safety.process(obs, raw)
                result = self.backend.send_action(safe)
                if not result.accepted:
                    self.tasks.status = TaskStatus.FAILURE
                    break
                steps += 1
                obs = self.backend.get_observation()
                success = self.detector.is_success(obs)
                if self.logger:
                    self.logger.write(
                        {
                            "task_id": task.task_id,
                            "observation_age_ms": (time.time_ns() - obs.timestamp_ns) / 1e6,
                            "raw_policy_action": raw.joints_deg.tolist(),
                            "safe_action": safe.joints_deg.tolist(),
                            "clipped": raw.joints_deg.tolist() != safe.joints_deg.tolist(),
                            "command_id": result.command_id,
                            "backend": result.backend,
                            "success": success,
                        }
                    )
                if success:
                    self.tasks.status = TaskStatus.SUCCESS
            if self.tasks.status == TaskStatus.RUNNING:
                self.tasks.status = TaskStatus.FAILURE
            return RolloutResult(self.tasks.status, steps, self.backend.health().backend)
        finally:
            self.backend.disconnect()


def default_logger(directory: str | Path, backend_name: str) -> TrajectoryLogger:
    return TrajectoryLogger(Path(directory) / f"{backend_name}-trajectory.jsonl")
