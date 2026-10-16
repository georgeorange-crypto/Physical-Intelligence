import json
from pathlib import Path

from robot_vla.dataset.recorder import LeRobotDatasetRecorder


def validate_mock_dataset(repo_id: str, root: str | Path, expected_episodes: int) -> dict:
    dataset = LeRobotDatasetRecorder.reload(repo_id, root)
    provenance_path = Path(root) / "meta" / "simulation_provenance.json"
    if not provenance_path.is_file():
        raise ValueError("dataset provenance sidecar is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("data_source") != "simulation_mock":
        raise ValueError("dataset is not marked as mock simulation data")
    if provenance.get("eligible_for_real_policy_training") is not False:
        raise ValueError("mock data must be marked ineligible for real policy training")
    if dataset.num_episodes != expected_episodes or dataset.num_frames <= 0:
        raise ValueError("dataset episode/frame count does not match expected values")
    return {"episodes": dataset.num_episodes, "frames": dataset.num_frames, **provenance}
