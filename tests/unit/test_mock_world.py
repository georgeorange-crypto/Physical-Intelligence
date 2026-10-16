import numpy as np

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.contracts import RobotAction


def test_mock_action_changes_joint_state_and_rgb_frame():
    backend = MockBackend()
    backend.connect()
    before = backend.get_observation()
    result = backend.send_action(RobotAction(np.array([120, 80, 90, 90, 90, 120])))
    after = backend.get_observation()
    assert result.accepted
    assert not np.array_equal(before.joints_deg, after.joints_deg)
    assert not np.array_equal(before.image_rgb, after.image_rgb)
    assert after.image_rgb.shape == (480, 640, 3)
