from dataclasses import dataclass
from enum import Enum


class TaskStatus(str, Enum):  # noqa: UP042
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
