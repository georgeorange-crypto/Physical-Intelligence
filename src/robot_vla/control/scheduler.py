from robot_vla.contracts import ActionChunk, RobotAction


class ActionChunkScheduler:
    """Sequential scheduler; first release executes one action then observes again."""

    def __init__(self, execution_horizon: int = 1) -> None:
        if execution_horizon < 1:
            raise ValueError("execution_horizon must be at least one")
        self.execution_horizon = execution_horizon

    def select(self, chunk: ActionChunk) -> tuple[RobotAction, ...]:
        return chunk.actions[: self.execution_horizon]
