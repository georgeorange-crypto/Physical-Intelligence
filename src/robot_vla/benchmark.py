"""Small, hardware-free policy inference benchmark on the deterministic mock task."""

import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.contracts import TaskSpec, TaskStatus
from robot_vla.policy.base import PolicyAdapter
from robot_vla.safety.controller import SafetyController
from robot_vla.task.success_detector import RedToBlueSuccessDetector


def run_mock_benchmark(
    policy_factory: Callable[[], PolicyAdapter],
    *,
    policy_name: str,
    checkpoint: str,
    episodes: int = 3,
    max_steps: int = 120,
    output_dir: str | Path = "logs/benchmarks",
) -> Path:
    """Run repeated model inference episodes, recording task and latency metrics."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")

    policy = policy_factory()
    task = TaskSpec("red_to_blue", "Place the red block in the blue area.")
    # Learned policy inference can exceed the normal 500 ms freshness window.
    # Relax only that timing gate in this mock-only evaluation; keep joint and rate limits.
    mock_safety_observation_timeout_ms = 10_000
    results: list[dict] = []
    all_latencies_ms: list[float] = []

    for episode in range(episodes):
        backend = MockBackend()
        safety = SafetyController(observation_timeout_ms=mock_safety_observation_timeout_ms)
        detector = RedToBlueSuccessDetector()
        policy.reset(task)
        result = {
            "episode": episode + 1,
            "status": TaskStatus.FAILURE.value,
            "steps": 0,
            "safety_rejections": 0,
            "error": None,
            "inference_latencies_ms": [],
        }
        backend.connect()
        try:
            observation = backend.get_observation()
            for _ in range(max_steps):
                start = time.perf_counter()
                try:
                    action = policy.predict(observation, task).actions[0]
                finally:
                    latency = (time.perf_counter() - start) * 1000
                    result["inference_latencies_ms"].append(latency)
                    all_latencies_ms.append(latency)
                try:
                    safe_action = safety.process(observation, action)
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    result["safety_rejections"] += 1
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    break
                action_result = backend.send_action(safe_action)
                if not action_result.accepted:
                    result["error"] = action_result.message or "mock backend rejected action"
                    break
                result["steps"] += 1
                observation = backend.get_observation()
                if detector.is_success(observation):
                    result["status"] = TaskStatus.SUCCESS.value
                    break
            else:
                result["error"] = "max_steps reached"
        except Exception as exc:  # retain a failed episode and continue the benchmark
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            backend.disconnect()
        results.append(result)

    successful_steps = [r["steps"] for r in results if r["status"] == TaskStatus.SUCCESS.value]
    summary = {
        "episodes": episodes,
        "successes": len(successful_steps),
        "success_rate": len(successful_steps) / episodes,
        "median_success_steps": statistics.median(successful_steps)
        if successful_steps
        else None,
        "total_safety_rejections": sum(r["safety_rejections"] for r in results),
        "inference_count": len(all_latencies_ms),
        "inference_mean_ms": statistics.mean(all_latencies_ms) if all_latencies_ms else None,
        "inference_p50_ms": _percentile(all_latencies_ms, 50),
        "inference_p95_ms": _percentile(all_latencies_ms, 95),
    }
    report = {
        "benchmark": "mock_red_to_blue_policy_inference",
        "data_source": "simulation_mock",
        "hardware_used": False,
        "policy": policy_name,
        "checkpoint": checkpoint,
        "seed_policy": "fixed identical initial scene per episode",
        "task": "red_to_blue",
        "max_steps": max_steps,
        "mock_safety_observation_timeout_ms": mock_safety_observation_timeout_ms,
        "summary": summary,
        "episodes": results,
        "limitations": [
            "MockWorld is a simplified deterministic image-to-joint simulator.",
            "Mock benchmark permits 10 s observation age; the normal rollout gate is 500 ms.",
            "Success does not establish real-robot task competence or sim-to-real transfer.",
            "Mock simulation data is not eligible for real policy training.",
        ],
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{policy_name}_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    return float(sorted(values)[round((len(values) - 1) * percentile / 100)])
