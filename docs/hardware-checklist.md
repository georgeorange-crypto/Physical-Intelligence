# Hardware connection checklist (future only)

No hardware was connected for this implementation. Real actions remain rejected until both a calibration file exists and `physical_calibration_verified` is true.

1. Verify SSH and network reachability.
2. Query `/health` only.
3. Verify camera streaming only.
4. Read joint feedback only.
5. Confirm continuous feedback stability.
6. Perform physical calibration and record verified safe ranges.
7. Send a very small single-joint action under direct observation.
8. Send a known coordinated six-joint action.
9. Connect laptop RpcBackend through an SSH tunnel.
10. Connect the LeRobot Robot adapter.
11. Run manual or scripted closed-loop tests.
12. Record real demonstrations into `datasets/real/`.
13. Train and evaluate ACT.
14. Only then evaluate SmolVLA outputs under a separately approved real-hardware procedure.

The mock driver envelopes are software bounds only. They do not assert physical limits or a neutral angle.
