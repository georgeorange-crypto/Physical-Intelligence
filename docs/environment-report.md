# Environment report

Recorded 2026-10-11 on Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU, driver 566.26 (reported driver CUDA capability 12.7).

| Interpreter | PyTorch | CUDA runtime | `torch.cuda.is_available()` | LeRobot |
|---|---|---|---|---|
| Python 3.10.11 (default machine interpreter) | 2.1.2+cu121 | 12.1 | `True`; RTX 4060 Laptop GPU | Not supported by pinned LeRobot 0.6.1 |
| Python 3.12.7 (`robot-vla/.venv`, isolated) | 2.7.1+cu126 | 12.6 | `True`; NVIDIA GeForce RTX 4060 Laptop GPU | 0.6.1; full optional SmolVLA dependencies installed |

The system Python 3.10 CUDA installation was retained untouched. The project `.venv` has `include-system-site-packages = false`; its dependencies pass `pip check`. It uses the CUDA 12.6 Torch build with the installed 12.7-capable NVIDIA driver. The LeRobot plugin factory, dataset v3, and full mock integrations passed in this isolated environment.

The isolated Python 3.12 environment passed 25 ordinary tests and one opt-in ACT model test. Python 3.10.11 passed 21 tests; four optional LeRobot integration tests were skipped because LeRobot is not installed in that interpreter. SmolVLA `lerobot/smolvla_base` completed a CUDA forward smoke with a finite 50×6 action chunk and 934.3 MiB peak allocated memory. ACT generated a temporary LeRobot v3 mock episode, trained one optimizer step on 16 sequential synthetic frames, then returned a finite 16×6 action chunk with 250.5 MiB peak allocated memory. The latest JSON reports are in `logs/model_smoke/`. These results verify inference/training plumbing and tensor compatibility, not learned task performance or physical hardware behavior. The SmolVLA smoke aliases the same synthetic camera frame to all camera views declared by that public checkpoint; ACT's sample is explicitly marked `simulation_mock` and not eligible for real policy training.

```powershell
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CUDA unavailable')"
```
