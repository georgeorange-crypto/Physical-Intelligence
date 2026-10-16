from pathlib import Path

from robot_vla.backend.mock_backend import MockBackend
from robot_vla.contracts import TaskStatus
from robot_vla.dataset.recorder import LeRobotDatasetRecorder
from robot_vla.policy.mock_policy import MockPolicy
from robot_vla.safety.controller import SafetyController
from robot_vla.task.registry import TaskRegistry
from robot_vla.task.success_detector import RedToBlueSuccessDetector


def record_mock_episodes(
    output: str | Path = "datasets/mock/red_to_blue",
    repo_id: str = "local/hiwonder_mock_red_to_blue",
    episodes: int = 3,
    max_steps: int = 120,
) -> tuple[int, int]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    recorder = LeRobotDatasetRecorder(repo_id, output, fps=10)
    task = TaskRegistry().resolve("red_to_blue")
    total_frames = 0
    try:
        for _ in range(episodes):
            backend = MockBackend()
            policy = MockPolicy()
            safety = SafetyController()
            detector = RedToBlueSuccessDetector()
            policy.reset(task)
            backend.connect()
            status = TaskStatus.RUNNING
            try:
                for _step in range(max_steps):
                    obs = backend.get_observation()
                    if detector.is_success(obs):
                        status = TaskStatus.SUCCESS
                        break
                    raw = policy.predict(obs, task).actions[0]
                    safe = safety.process(obs, raw)
                    result = backend.send_action(safe)
                    if not result.accepted:
                        status = TaskStatus.FAILURE
                        break
                    recorder.add_frame(obs, safe.joints_deg, task.instruction)
                    total_frames += 1
                else:
                    status = TaskStatus.FAILURE
                if status != TaskStatus.SUCCESS:
                    raise RuntimeError(f"mock episode did not succeed: {status.value}")
                recorder.save_episode()
            finally:
                backend.disconnect()
        recorder.finalize()
    except BaseException:
        # Finalize interrupted data so it cannot be mistaken for a complete episode.
        recorder.finalize()
        raise

    reloaded = LeRobotDatasetRecorder.reload(repo_id, output)
    if reloaded.num_episodes != episodes or reloaded.num_frames == 0:
        raise RuntimeError(
            f"dataset reload mismatch: episodes={reloaded.num_episodes}, "
            f"frames={reloaded.num_frames}"
        )
    return reloaded.num_episodes, reloaded.num_frames
