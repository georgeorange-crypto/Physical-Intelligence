import cv2
import numpy as np

from robot_vla.contracts import RobotObservation


class RedToBlueSuccessDetector:
    def __init__(self, stable_frames: int = 3) -> None:
        self.stable_frames = stable_frames
        self._count = 0

    @staticmethod
    def _mask(
        rgb: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]
    ) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

    def is_success(self, observation: RobotObservation) -> bool:
        red = self._mask(observation.image_rgb, (0, 100, 80), (10, 255, 255)) | self._mask(
            observation.image_rgb, (170, 100, 80), (180, 255, 255)
        )
        blue = self._mask(observation.image_rgb, (95, 80, 50), (135, 255, 255))
        moments = cv2.moments(red)
        blue_points = cv2.findNonZero(blue)
        inside = False
        if moments["m00"] and blue_points is not None and observation.joints_deg[5] >= 100:
            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]
            bx, by, bw, bh = cv2.boundingRect(blue_points)
            inside = bx <= x <= bx + bw and by <= y <= by + bh
        self._count = self._count + 1 if inside else 0
        return self._count >= self.stable_frames
