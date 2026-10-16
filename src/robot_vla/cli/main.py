import argparse
import time

import uvicorn

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.backend.rpc_backend import RpcBackend
from robot_vla.control.rollout import RolloutController, default_logger
from robot_vla.policy.mock_policy import MockPolicy
from robot_vla.safety.controller import SafetyController


def _backend(kind: str, base_url: str):
    if kind == "mock":
        return MockBackend()
    return RpcBackend(base_url)


def main() -> None:
    parser = argparse.ArgumentParser(prog="robot-vla")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("mock-server", help="run the local mock armd RPC server")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    run = sub.add_parser("run", help="run one synchronous task rollout")
    run.add_argument("--backend", choices=("mock", "rpc"), default="mock")
    run.add_argument("--policy", choices=("mock", "act", "smolvla"), default="mock")
    run.add_argument("--task", default="red_to_blue")
    run.add_argument("--base-url", default="http://127.0.0.1:8765")
    run.add_argument("--log-dir", default="logs")
    run.add_argument("--max-steps", type=int, default=120)
    inspect = sub.add_parser("inspect", help="display current backend state")
    inspect.add_argument("--backend", choices=("mock", "rpc"), default="mock")
    inspect.add_argument("--base-url", default="http://127.0.0.1:8765")
    inspect.add_argument("--once", action="store_true")
    smoke = sub.add_parser("model-smoke-test", help="opt-in LeRobot policy load/inference check")
    smoke.add_argument("--policy", choices=("act", "smolvla"), required=True)
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--checkpoint", default=None)
    benchmark = sub.add_parser(
        "benchmark", help="evaluate learned-policy inference on the hardware-free mock task"
    )
    benchmark.add_argument("--policy", choices=("act", "smolvla"), required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--checkpoint", default=None)
    benchmark.add_argument("--episodes", type=int, default=3)
    benchmark.add_argument("--max-steps", type=int, default=120)
    benchmark.add_argument("--output-dir", default="logs/benchmarks")
    record = sub.add_parser("record-mock", help="record and reload LeRobot v3 mock episodes")
    record.add_argument("--episodes", type=int, default=3)
    record.add_argument("--output", default="datasets/mock/red_to_blue")
    record.add_argument("--repo-id", default="local/hiwonder_mock_red_to_blue")
    args = parser.parse_args()

    if args.command == "mock-server":
        uvicorn.run("robot_vla.rpc.mock_armd:app", host=args.host, port=args.port, log_level="info")
    elif args.command == "run":
        backend = _backend(args.backend, args.base_url)
        if args.policy != "mock":
            raise SystemExit(
                "ACT/SmolVLA rollout needs a trained checkpoint; run model-smoke-test first"
            )
        controller = RolloutController(
            backend,
            MockPolicy(),
            SafetyController(),
            logger=default_logger(args.log_dir, args.backend),
            max_steps=args.max_steps,
        )
        result = controller.run(args.task)
        print(f"status={result.status.value} steps={result.steps} backend={result.backend}")
        if result.status.value != "success":
            raise SystemExit(1)
    elif args.command == "inspect":
        backend = _backend(args.backend, args.base_url)
        backend.connect()
        previous_timestamp = None
        try:
            while True:
                obs = backend.get_observation()
                camera_fps = (
                    0.0
                    if previous_timestamp is None
                    else 1e9 / max(obs.timestamp_ns - previous_timestamp, 1)
                )
                previous_timestamp = obs.timestamp_ns
                print(
                    f"health={backend.health()} camera_fps={camera_fps:.2f} "
                    f"timestamp_ns={obs.timestamp_ns} "
                    f"joints_deg={obs.joints_deg.tolist()}"
                )
                if args.once:
                    break
                time.sleep(0.5)
        finally:
            backend.disconnect()
    elif args.command == "model-smoke-test":
        from robot_vla.policy.lerobot_adapter import run_model_smoke_test

        run_model_smoke_test(args.policy, args.device, args.checkpoint)
    elif args.command == "benchmark":
        import torch

        from robot_vla.benchmark import run_mock_benchmark
        from robot_vla.policy.lerobot_adapter import ACTPolicyAdapter, SmolVLAPolicyAdapter

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
        checkpoint = args.checkpoint
        if args.policy == "act" and checkpoint is None:
            raise SystemExit("ACT benchmark requires --checkpoint; train a policy first")
        checkpoint = checkpoint or "lerobot/smolvla_base"
        adapter_type = SmolVLAPolicyAdapter if args.policy == "smolvla" else ACTPolicyAdapter
        report_path = run_mock_benchmark(
            lambda: adapter_type(checkpoint, args.device),
            policy_name=args.policy,
            checkpoint=checkpoint,
            episodes=args.episodes,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
        print(
            f"benchmark=mock_red_to_blue_policy_inference policy={args.policy} "
            f"successes={summary['successes']}/{summary['episodes']} "
            f"success_rate={summary['success_rate']:.3f} "
            f"inference_mean_ms={summary['inference_mean_ms']} report={report_path}"
        )
    elif args.command == "record-mock":
        from robot_vla.dataset.record_mock import record_mock_episodes

        episodes, frames = record_mock_episodes(args.output, args.repo_id, args.episodes)
        print(
            f"data_source=simulation_mock eligible_for_real_policy_training=false "
            f"episodes={episodes} frames={frames}"
        )


if __name__ == "__main__":
    main()
