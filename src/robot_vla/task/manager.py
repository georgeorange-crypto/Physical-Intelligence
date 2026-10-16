import time

from robot_vla.contracts import TaskSpec, TaskStatus


class TaskManager:
    def __init__(self) -> None:
        self.status = TaskStatus.IDLE
        self.started_at = 0.0
        self.task: TaskSpec | None = None

    def start(self, task: TaskSpec) -> TaskSpec:
        self.task = task
        self.started_at = time.monotonic()
        self.status = TaskStatus.RUNNING
        return task

    def cancel(self) -> None:
        if self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.CANCELLED

    def check_timeout(self, timeout_s: float) -> bool:
        if self.status == TaskStatus.RUNNING and time.monotonic() - self.started_at > timeout_s:
            self.status = TaskStatus.TIMEOUT
        return self.status == TaskStatus.TIMEOUT
