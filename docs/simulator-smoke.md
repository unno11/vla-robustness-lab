# LIBERO simulator diagnostic

This entry point is a bounded simulator smoke check, not a policy rollout or a benchmark. It uses upstream LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` without local source edits.

## Protocol

- Task: the first `libero_spatial` task in the upstream task map (black bowl between plate and ramekin).
- Seed: 0; initialization: seeded simulator reset. This is **not** an official benchmark initial-state index.
- Observations: `agentview_image` and `robot0_eye_in_hand_image`, 256 x 256 x 3 uint8. Save raw arrays and vertically flipped PNG previews separately.
- Budget: at most five zero-arm/open-gripper diagnostic actions, stop earlier on native done; no policy.
- Pass: both nonblank observations saved, diagnostic step loop completed, environment explicitly closed, process exits 0.
- Failure: import, initialization, image validation, stepping or close error. Retain the run directory and traceback. An external 180-second timeout is infrastructure failure, not task failure.

Run in an isolated Python 3.10 environment after installing simulator dependencies:

```bash
python scripts/libero_smoke.py --libero-root /path/to/LIBERO --output /path/to/new-run-directory --gpu 0
```

Check available GPU memory/load before choosing a device. The script creates its own `LIBERO_CONFIG_PATH`; it does not overwrite the user's global LIBERO configuration. `result.json` records the upstream commit, script and task fingerprints, observation metadata and close status. This protocol is frozen for this diagnostic; benchmark and policy integration require separate initial-state and interface verification.

## Existing dataset audit (2026-10-28)

A pre-existing LeRobot v2.1 LIBERO-10 directory was checked read-only. Its metadata declares 379 episodes, 101,469 frames and 758 camera videos. All 379 Parquet files decode, totaling 101,469 rows. Only 51 video files are present: 707 expected videos are missing. Existing container signatures are valid. Video decoding was attempted with cached OpenCV 4.11.0.86; its FFmpeg AV1 decoder failed, so video integrity remains unverified rather than proven corrupt.

This directory is not accepted as a complete visual training dataset. No subset was selected for training, and no training was started. Simulator initialization uses the pinned repository's task definitions and assets, not demonstration videos. The audit therefore does not change the simulator protocol or any evaluation results. No dataset has been republished.
