import pytest


@pytest.mark.lerobot
def test_three_mock_episodes_write_reload_with_provenance(tmp_path):
    pytest.importorskip("lerobot")
    try:
        import cv2  # noqa: F401
        import pyarrow  # noqa: F401

        import datasets  # noqa: F401
    except ImportError:
        pytest.skip("install robot-vla[lerobot] to run the LeRobotDataset v3 integration")

    from robot_vla.dataset.record_mock import record_mock_episodes

    episode_count, frame_count = record_mock_episodes(
        tmp_path / "dataset", "local/test_mock_episodes", episodes=3
    )
    assert episode_count == 3
    assert frame_count > 0
    assert (tmp_path / "dataset" / "meta" / "simulation_provenance.json").is_file()
