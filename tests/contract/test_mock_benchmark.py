import json

import pytest

from robot_vla.benchmark import run_mock_benchmark
from robot_vla.policy.mock_policy import MockPolicy


def test_mock_benchmark_records_reproducible_task_metrics(tmp_path):
    report_path = run_mock_benchmark(
        MockPolicy,
        policy_name="mock_reference",
        checkpoint="builtin_mock_policy",
        episodes=2,
        output_dir=tmp_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["hardware_used"] is False
    assert report["data_source"] == "simulation_mock"
    assert report["summary"]["successes"] == 2
    assert [episode["steps"] for episode in report["episodes"]] == [45, 45]
    assert report["summary"]["inference_count"] == 90


def test_mock_benchmark_rejects_invalid_episode_count(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_mock_benchmark(
            MockPolicy,
            policy_name="mock_reference",
            checkpoint="builtin_mock_policy",
            episodes=0,
            output_dir=tmp_path,
        )
