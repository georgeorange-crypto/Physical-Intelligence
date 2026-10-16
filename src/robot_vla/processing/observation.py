from typing import Any

import numpy as np

from robot_vla.contracts import RobotObservation


def observation_to_le_robot(observation: RobotObservation) -> dict[str, Any]:
    return {
        "observation.state": observation.joints_deg.astype(np.float32, copy=True),
        "observation.images.image": observation.image_rgb.copy(),
    }
