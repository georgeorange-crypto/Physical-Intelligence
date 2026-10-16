import json
from pathlib import Path

import numpy as np

from robot_vla.contracts import RobotObservation
from robot_vla.dataset.schema import validate_dataset_frame


class LeRobotDatasetRecorder:
    """Thin LeRobotDataset v3 writer; importing LeRobot remains optional."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int = 10,
        image_size: tuple[int, int] = (480, 640),
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "install the optional `lerobot` extra to record LeRobotDataset v3"
            ) from exc
        self.dataset_type = LeRobotDataset
        h, w = image_size
        joints = [f"joint_{i}.pos" for i in range(1, 7)]
        features = {
            "observation.images.image": {
                "dtype": "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {"dtype": "float32", "shape": (6,), "names": joints},
            "action": {"dtype": "float32", "shape": (6,), "names": joints},
        }
        self.root = Path(root)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=self.root,
            fps=fps,
            robot_type="hiwonder_arm",
            features=features,
            use_videos=True,
        )

    def add_frame(self, observation: RobotObservation, action: np.ndarray, task: str) -> None:
        frame = {
            "observation.images.image": observation.image_rgb,
            "observation.state": observation.joints_deg.astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": task,
        }
        validate_dataset_frame(frame, observation.image_rgb.shape)
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        self.dataset.save_episode()

    def finalize(
        self,
        *,
        data_source: str = "simulation_mock",
        eligible_for_real_policy_training: bool = False,
    ) -> None:
        self.dataset.finalize()
        provenance = {
            "data_source": data_source,
            "eligible_for_real_policy_training": eligible_for_real_policy_training,
            "robot_type": "hiwonder_arm",
        }
        path = self.root / "meta" / "simulation_provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    @classmethod
    def reload(cls, repo_id: str, root: str | Path):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError("install the optional `lerobot` extra to reload datasets") from exc
        return LeRobotDataset(repo_id, root=root)


class MockDatasetRecorder:
    """Always-available JSONL/PNG episode artifact for offline smoke workflows."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict] = []

    def add_frame(self, observation: RobotObservation, action: np.ndarray, task: str) -> None:
        self.frames.append(
            {
                "timestamp_ns": observation.timestamp_ns,
                "joints_deg": observation.joints_deg.tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "task": task,
                "image_rgb": observation.image_rgb.tolist(),
            }
        )

    def save_episode(self, episode_index: int = 0) -> Path:
        path = self.directory / f"episode_{episode_index:04d}.json"
        payload = {
            "data_source": "simulation_mock",
            "eligible_for_real_policy_training": False,
            "frames": self.frames,
        }
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        self.frames.clear()
        return path
