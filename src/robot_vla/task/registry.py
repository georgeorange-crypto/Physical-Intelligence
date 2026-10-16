from robot_vla.contracts import TaskSpec


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks = {
            "red_to_blue": TaskSpec("red_to_blue", "Place the red block in the blue area.")
        }
        self._aliases = {
            "把红色方块放进蓝色区域": "red_to_blue",
            "把红块放到蓝区": "red_to_blue",
        }

    def resolve(self, command: str) -> TaskSpec:
        task_id = self._aliases.get(command, command)
        if task_id not in self._tasks:
            raise KeyError(f"unknown task: {command}")
        return self._tasks[task_id]
