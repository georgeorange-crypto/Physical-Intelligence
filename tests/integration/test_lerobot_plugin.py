import pytest


def test_custom_lerobot_robot_uses_mock_backend_and_safety():
    pytest.importorskip("lerobot")
    pytest.importorskip("cv2")
    plugin = pytest.importorskip("lerobot_robot_hiwonder_arm")

    arm = plugin.HiwonderArm(plugin.HiwonderArmConfig())
    try:
        arm.connect()
        observation = arm.get_observation()
        assert observation["observation.images.image"].shape == (480, 640, 3)
        assert all(f"joint_{i}.pos" in observation for i in range(1, 7))
        action = {f"joint_{i}.pos": 90.0 for i in range(1, 7)}
        safe_action = arm.send_action(action)
        assert safe_action["joint_6.pos"] == 115.0
        assert arm.is_connected and arm.is_calibrated
    finally:
        arm.disconnect()


def test_custom_robot_is_registered_with_lerobot_factory():
    pytest.importorskip("lerobot")
    plugin = pytest.importorskip("lerobot_robot_hiwonder_arm")
    from lerobot.robots.utils import make_robot_from_config

    robot = make_robot_from_config(plugin.HiwonderArmConfig())
    assert isinstance(robot, plugin.HiwonderArm)
    assert robot.name == "hiwonder_arm"


def test_rpc_backend_in_mock_mode_does_not_require_physical_calibration():
    pytest.importorskip("lerobot")
    plugin = pytest.importorskip("lerobot_robot_hiwonder_arm")
    from robot_vla.backend.mock_backend import MockBackend

    config = plugin.HiwonderArmConfig(mode="mock", backend_type="rpc")
    arm = plugin.HiwonderArm(config)
    arm.backend = MockBackend()
    try:
        arm.connect()
        action = {f"joint_{i}.pos": 90.0 for i in range(1, 7)}
        assert arm.send_action(action)["joint_6.pos"] == 115.0
    finally:
        arm.disconnect()
