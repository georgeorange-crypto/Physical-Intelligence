import pytest
import torch

from robot_vla.policy.lerobot_adapter import run_act_synthetic_smoke_test


@pytest.mark.model
def test_act_six_joint_synthetic_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_act_synthetic_smoke_test(device)
