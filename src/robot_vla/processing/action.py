import numpy as np

from robot_vla.contracts import ActionChunk


def action_to_le_robot(chunk: ActionChunk) -> np.ndarray:
    result = np.stack([action.joints_deg for action in chunk.actions]).astype(np.float32)
    if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all():
        raise ValueError("action chunk must have shape (H, 6) and finite values")
    return result
