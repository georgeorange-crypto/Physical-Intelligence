import cv2
import numpy as np

from robot_vla.mock.world import MockWorld


def render_rgb(world: MockWorld) -> np.ndarray:
    """Render the mock scene as an RGB uint8 camera frame."""
    image = np.full((world.height, world.width, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (420, 180), (520, 280), (30, 80, 235), -1)
    cv2.rectangle(image, (420, 180), (520, 280), (10, 25, 120), 3)
    cx, cy = np.rint(world.cube_xy).astype(int)
    cv2.rectangle(image, (cx - 16, cy - 16), (cx + 16, cy + 16), (235, 25, 25), -1)
    gx, gy = np.rint(world.gripper_xy).astype(int)
    cv2.circle(image, (gx, gy), 12, (30, 30, 30), -1)
    cv2.circle(image, (gx, gy), 5, (220, 220, 220), -1)
    return image
