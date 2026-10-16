import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from robot_vla.contracts import ActionChunk, RobotAction, RobotObservation, TaskSpec
from robot_vla.policy.base import PolicyAdapter

UTC_TZ = timezone(timedelta(0))

JOINT_NAMES = [f"joint_{i}.pos" for i in range(1, 7)]


class LeRobotPolicyAdapter(PolicyAdapter):
    """Policy adapter that uses the checkpoint's serialized input/output pipelines."""

    policy_class_path: str
    policy_module: str

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        try:
            module = __import__(self.policy_module, fromlist=[self.policy_class_path])
            policy_class = getattr(module, self.policy_class_path)
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "install the optional `lerobot` extra to use learned policies"
            ) from exc

        self.checkpoint = checkpoint
        self.device = device
        self.policy = policy_class.from_pretrained(checkpoint).to(device).eval()
        self.input_features = self.policy.config.input_features
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _frame(self, observation: RobotObservation, task: TaskSpec) -> dict:
        import torch
        from lerobot.policies.utils import build_inference_frame

        raw_observation = {
            f"joint_{i}.pos": float(observation.joints_deg[i - 1]) for i in range(1, 7)
        }
        dataset_features = {}
        for key, feature in self.input_features.items():
            feature_type = getattr(feature.type, "value", feature.type)
            if key == "observation.state":
                if tuple(feature.shape) != (6,):
                    raise ValueError(
                        f"checkpoint state must have six joints, got {tuple(feature.shape)}"
                    )
                dataset_features[key] = {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": JOINT_NAMES,
                }
            elif feature_type == "VISUAL" and key.startswith("observation.images."):
                camera_name = key.removeprefix("observation.images.")
                dataset_features[key] = {
                    "dtype": "video",
                    "shape": observation.image_rgb.shape,
                    "names": ["height", "width", "channel"],
                }
                # The current RobotObservation contract carries one RGB image. For a
                # checkpoint with multiple views, use that synthetic view for each input
                # during smoke validation; real multi-camera inference needs a richer contract.
                raw_observation[camera_name] = observation.image_rgb

        if not any(key.startswith("observation.images.") for key in dataset_features):
            raise ValueError("checkpoint does not declare a supported visual input")
        return build_inference_frame(
            observation=raw_observation,
            ds_features=dataset_features,
            device=torch.device(self.device),
            task=task.instruction,
            robot_type="hiwonder_arm",
        )

    def predict(self, observation: RobotObservation, task: TaskSpec) -> ActionChunk:
        import torch

        frame = self.preprocessor(self._frame(observation, task))
        with torch.inference_mode():
            if hasattr(self.policy, "predict_action_chunk"):
                prediction = self.policy.predict_action_chunk(frame)
            else:
                prediction = self.policy.select_action(frame)
            prediction = self.postprocessor(prediction)

        if isinstance(prediction, dict):
            if "action" not in prediction:
                raise ValueError(
                    f"postprocessor did not return an action tensor: {prediction.keys()}"
                )
            prediction = prediction["action"]
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.as_tensor(prediction)
        prediction = prediction.detach().float().cpu()
        if prediction.ndim == 1:
            prediction = prediction.unsqueeze(0)
        elif prediction.ndim == 3:
            prediction = prediction[0]
        if prediction.ndim != 2 or prediction.shape[1] != 6:
            raise ValueError(f"policy output must have shape (H, 6), got {tuple(prediction.shape)}")
        actions = np.asarray(prediction, dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("policy produced NaN or infinite joint targets")
        return ActionChunk(tuple(RobotAction(row) for row in actions))


class ACTPolicyAdapter(LeRobotPolicyAdapter):
    """Lazy adapter for a six-joint LeRobot ACT checkpoint."""

    policy_class_path = "ACTPolicy"
    policy_module = "lerobot.policies.act"


class SmolVLAPolicyAdapter(LeRobotPolicyAdapter):
    """Lazy adapter for a six-joint LeRobot SmolVLA checkpoint."""

    policy_class_path = "SmolVLAPolicy"
    policy_module = "lerobot.policies.smolvla"


def run_model_smoke_test(policy_name: str, device: str, checkpoint: str | None) -> None:
    import torch

    started = time.perf_counter()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    if policy_name == "act" and checkpoint is None:
        run_act_synthetic_smoke_test(device)
        return
    if checkpoint is None:
        checkpoint = "lerobot/smolvla_base"

    from robot_vla.backend.mock_backend import MockBackend

    backend = MockBackend()
    backend.connect()
    observation = backend.get_observation()
    task = TaskSpec("red_to_blue", "Place the red block in the blue area.")
    adapter: PolicyAdapter = (
        SmolVLAPolicyAdapter(checkpoint, device)
        if policy_name == "smolvla"
        else ACTPolicyAdapter(checkpoint, device)
    )
    adapter.reset(task)
    chunk = adapter.predict(observation, task)
    backend.disconnect()
    report = (
        f"policy={policy_name} device={device} action_chunk={len(chunk.actions)} "
        f"finite=true checkpoint={checkpoint}"
    )
    if device.startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        report += f" cuda_peak_memory_mb={peak_mb:.1f}"
    report_path = _write_smoke_report(
        {
            "policy": policy_name,
            "checkpoint": checkpoint,
            "device": device,
            "action_shape": [len(chunk.actions), 6],
            "finite": True,
            "duration_s": round(time.perf_counter() - started, 3),
            "cuda_peak_memory_mb": peak_mb if device.startswith("cuda") else None,
            "cuda_device_name": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else None,
        }
    )
    report += f" report={report_path}"
    print(report)


def run_act_synthetic_smoke_test(device: str) -> None:
    """Train and infer one step using a temporary synthetic LeRobot v3 episode."""
    from tempfile import TemporaryDirectory
    from uuid import uuid4

    import torch
    import torch.nn.functional as torch_functional
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig

    from robot_vla.dataset.record_mock import record_mock_episodes

    started = time.perf_counter()
    config = ACTConfig(
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        device=device,
        pretrained_backbone_weights=None,
        chunk_size=16,
        n_action_steps=1,
        dim_model=128,
        n_heads=4,
        dim_feedforward=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
        latent_dim=16,
        n_vae_encoder_layers=2,
    )
    policy = ACTPolicy(config).to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    dataset_frames = config.chunk_size
    with TemporaryDirectory(prefix="robot-vla-act-smoke-") as temp_dir:
        output = Path(temp_dir) / "mock_dataset"
        repo_id = f"local/act_smoke_{uuid4().hex}"
        record_mock_episodes(output, repo_id, episodes=1)
        dataset = LeRobotDataset(repo_id, root=output)
        samples = [dataset[index] for index in range(dataset_frames)]
        image_batch = torch.stack([sample["observation.images.image"] for sample in samples])
        image_batch = torch_functional.interpolate(
            image_batch[:1].to(device=device, dtype=torch.float32),
            size=(96, 96),
            mode="bilinear",
            align_corners=False,
        )
        state_batch = torch.stack([sample["observation.state"] for sample in samples[:1]]).to(
            device=device
        )
        action_chunk = (
            torch.stack([sample["action"] for sample in samples]).unsqueeze(0).to(device=device)
        )
        batch = {
            "observation.images.image": image_batch,
            "observation.state": state_batch,
            "action": action_chunk,
            "action_is_pad": torch.zeros((1, dataset_frames), dtype=torch.bool, device=device),
        }
        policy.train()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.optimizer_lr)
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError("ACT synthetic training loss is not finite")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        policy.eval()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)

    if actions.shape != (1, config.chunk_size, 6):
        raise ValueError(
            f"ACT output must be (1, {config.chunk_size}, 6), got {tuple(actions.shape)}"
        )
    if not torch.isfinite(actions).all():
        raise ValueError("ACT produced NaN or infinite joint targets")
    report = (
        f"policy=act device={device} action_chunk={actions.shape[1]}x{actions.shape[2]} "
        "finite=true checkpoint=untrained_synthetic training_source=simulation_mock"
    )
    if device.startswith("cuda"):
        report += f" cuda_peak_memory_mb={torch.cuda.max_memory_allocated(device) / (1024**2):.1f}"
    report_path = _write_smoke_report(
        {
            "policy": "act",
            "checkpoint": "untrained_synthetic",
            "training_source": "simulation_mock",
            "training_loss": float(loss.detach().cpu()),
            "training_frames": dataset_frames,
            "device": device,
            "action_shape": list(actions.shape[1:]),
            "finite": True,
            "duration_s": round(time.perf_counter() - started, 3),
            "cuda_peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.startswith("cuda")
            else None,
            "cuda_device_name": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else None,
        }
    )
    report += f" report={report_path}"
    print(report)


def _write_smoke_report(result: dict) -> str:
    result["timestamp_utc"] = datetime.now(UTC_TZ).isoformat()
    report_dir = Path("logs/model_smoke")
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(UTC_TZ).strftime("%Y%m%dT%H%M%S_%fZ")
    path = report_dir / f"{result['policy']}_{filename}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
