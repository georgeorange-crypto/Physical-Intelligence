from abc import ABC, abstractmethod

from robot_vla.contracts import ActionChunk, RobotObservation, TaskSpec


class PolicyAdapter(ABC):
    @abstractmethod
    def reset(self, task: TaskSpec) -> None: ...

    @abstractmethod
    def predict(self, observation: RobotObservation, task: TaskSpec) -> ActionChunk: ...
