from robot_vla.backend.mock_backend import MockBackend
from robot_vla.contracts import TaskStatus
from robot_vla.control.rollout import RolloutController
from robot_vla.policy.mock_policy import MockPolicy


def test_mock_policy_completes_red_to_blue_closed_loop():
    backend = MockBackend()
    result = RolloutController(backend, MockPolicy(), max_steps=100).run("red_to_blue")
    assert result.status == TaskStatus.SUCCESS
    assert result.steps < 100
    assert result.backend == "mock"
    assert backend.world.cube_xy[0] >= 420 and backend.world.cube_xy[0] <= 520
    assert backend.world.cube_xy[1] >= 180 and backend.world.cube_xy[1] <= 280
    assert not backend.world.attached
