import numpy as np


def validate_dataset_frame(frame: dict, image_shape: tuple[int, int, int]) -> None:
    required = {"observation.images.image", "observation.state", "action", "task"}
    missing = required - frame.keys()
    if missing:
        raise ValueError(f"dataset frame missing keys: {sorted(missing)}")
    image = np.asarray(frame["observation.images.image"])
    state = np.asarray(frame["observation.state"])
    action = np.asarray(frame["action"])
    if image.shape != image_shape or image.dtype != np.uint8:
        raise ValueError("dataset image must match declared HWC uint8 schema")
    if state.shape != (6,) or action.shape != (6,):
        raise ValueError("dataset state and action must each have six joints")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("dataset state/action must be finite")
    if not isinstance(frame["task"], str) or not frame["task"]:
        raise ValueError("dataset task instruction must be non-empty")
