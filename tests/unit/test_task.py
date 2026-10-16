import numpy as np

from robot_vla.contracts import RobotObservation
from robot_vla.mock.world import MockWorld
from robot_vla.task.registry import TaskRegistry
from robot_vla.task.success_detector import RedToBlueSuccessDetector


def test_task_registry_resolves_canonical_and_chinese_alias():
    registry = TaskRegistry()
    assert registry.resolve("把红色方块放进蓝色区域").task_id == "red_to_blue"
    assert registry.resolve("red_to_blue").instruction == "Place the red block in the blue area."


def test_success_detector_requires_release_and_stable_frames():
    world = MockWorld(cube_xy=np.array([470, 230], np.float32))
    image = world.render()
    detector = RedToBlueSuccessDetector(stable_frames=2)
    closed = RobotObservation(1, np.array([90, 90, 90, 90, 90, 30]), image)
    opened = RobotObservation(1, np.array([90, 90, 90, 90, 90, 120]), image)
    assert not detector.is_success(closed)
    assert not detector.is_success(opened)
    assert detector.is_success(opened)
