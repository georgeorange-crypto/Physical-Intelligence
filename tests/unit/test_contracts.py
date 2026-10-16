import numpy as np
import pytest

from robot_vla.contracts import RobotAction, RobotObservation


def test_observation_contract_shapes_and_types():
    obs = RobotObservation(1, np.zeros(6), np.zeros((48, 64, 3), dtype=np.uint8))
    assert obs.joints_deg.dtype == np.float32
    assert obs.image_rgb.dtype == np.uint8


@pytest.mark.parametrize(
    "joints,image",
    [
        ([0] * 5, np.zeros((10, 10, 3), dtype=np.uint8)),
        ([0, 0, 0, 0, 0, np.nan], np.zeros((10, 10, 3), dtype=np.uint8)),
        ([0] * 6, np.zeros((10, 10), dtype=np.uint8)),
        ([0] * 6, np.zeros((10, 10, 3), dtype=np.float32)),
    ],
)
def test_observation_rejects_invalid_data(joints, image):
    with pytest.raises(ValueError):
        RobotObservation(1, np.asarray(joints), image)


def test_action_rejects_nonfinite_and_invalid_duration():
    with pytest.raises(ValueError):
        RobotAction(np.array([0, 0, 0, 0, 0, np.inf]))
    with pytest.raises(ValueError):
        RobotAction(np.zeros(6), 0)
