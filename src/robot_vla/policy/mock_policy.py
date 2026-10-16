import cv2
import numpy as np

from robot_vla.contracts import ActionChunk, RobotAction, RobotObservation, TaskSpec
from robot_vla.policy.base import PolicyAdapter


class MockPolicy(PolicyAdapter):
    """Image-driven scripted policy; it never reads or mutates MockWorld state."""

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        self._phase = "pick"

    @staticmethod
    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        moments = cv2.moments(mask)
        if not moments["m00"]:
            raise RuntimeError("required object is not visible")
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    def predict(self, observation: RobotObservation, task: TaskSpec) -> ActionChunk:
        hsv = cv2.cvtColor(observation.image_rgb, cv2.COLOR_RGB2HSV)
        red = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255)) | cv2.inRange(
            hsv, (170, 100, 80), (180, 255, 255)
        )
        blue = cv2.inRange(hsv, (95, 80, 50), (135, 255, 255))
        cube = self._centroid(red)
        target = self._centroid(blue)
        h, w = observation.image_rgb.shape[:2]
        target_q1 = float(np.interp(target[0], [40, w - 40], [0, 180]))
        target_q2 = float(np.interp(target[1], [40, h - 40], [180, 0]))
        cube_q1 = float(np.interp(cube[0], [40, w - 40], [0, 180]))
        cube_q2 = float(np.interp(cube[1], [40, h - 40], [180, 0]))
        current = observation.joints_deg
        distance = float(np.linalg.norm(current[:2] - np.array([cube_q1, cube_q2])))
        carrying = current[5] <= 60 and distance < 4.0
        if self._phase == "pick" and carrying:
            self._phase = "carry"
        if self._phase == "carry":
            target_joints = current.copy()
            at_target = float(np.linalg.norm(current[:2] - np.array([target_q1, target_q2]))) < 4.0
            target_joints[0:2] = (target_q1, target_q2)
            if at_target:
                self._phase = "release"
            target_joints[5] = 120.0 if at_target else 35.0
        elif self._phase == "release":
            target_joints = current.copy()
            target_joints[0:2] = (target_q1, target_q2)
            target_joints[5] = 120.0
        else:
            target_joints = current.copy()
            target_joints[0:2] = (cube_q1, cube_q2)
            target_joints[5] = 35.0
        return ActionChunk((RobotAction(target_joints),))
