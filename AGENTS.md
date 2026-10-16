# Project rules

- Never connect to or command real hardware during laptop-first development.
- Hardware libraries belong only behind a backend; application code must not import `Arm_Lib`.
- Every policy action passes through `SafetyController` before a backend call.
- Real RPC actions require verified physical calibration; mock envelopes are not physical limits.
- Mock trajectories are simulation artifacts and must be labeled ineligible for real policy training.
- Keep model downloads and GPU inference behind explicit opt-in commands/markers.
- Run `ruff check .` and `pytest` for implementation phases; report unavailable platform checks honestly.
