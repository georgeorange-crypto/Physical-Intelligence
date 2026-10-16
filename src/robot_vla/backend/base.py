from abc import ABC, abstractmethod

from robot_vla.contracts import ActionResult, HealthStatus, RobotAction, RobotObservation


class RobotBackend(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def health(self) -> HealthStatus: ...

    @abstractmethod
    def get_observation(self) -> RobotObservation: ...

    @abstractmethod
    def send_action(self, action: RobotAction) -> ActionResult: ...
