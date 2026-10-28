# CPU environment and data-read validation

Validated on 2026-10-29 (Asia/Shanghai) with Python 3.10.12 and LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`. [CPU evidence](../results/cpu_validation.json) is separate from [overall experiment status](../results/status.json).

## Independent uv environment

The tested dependency inventory is [libero-cpu.snapshot.txt](../requirements/libero-cpu.snapshot.txt). It is a `uv pip freeze` snapshot, not a cross-platform hash lock or an OpenVLA inference environment. Dependencies were installed into a dedicated environment without modifying the older simulation project or global LIBERO settings. Additions resolving observed import failures were `termcolor==2.4.0` and upstream-required `future==0.18.2`.

A new installation can use uv and this snapshot, then must rerun validation:

```bash
uv venv --python 3.10.12 .venv-libero
uv pip install --python .venv-libero/bin/python -r requirements/libero-cpu.snapshot.txt
.venv-libero/bin/python scripts/libero_smoke.py --libero-root /path/to/pinned/LIBERO --output /path/to/new-cpu-run --mode physics
```

The physics mode sets `CUDA_VISIBLE_DEVICES` empty and `MUJOCO_GL=disable`, disables both rendering and camera observations, and calls upstream `ControlEnv` with seed 0. The tested entry point initialized the first Spatial task, executed five fixed diagnostic actions and closed normally (exit 0). The task was not successful, as expected for a dummy-action diagnostic. No policy was used; this is not a policy rollout or a benchmark score.

The default camera mode still requires a GPU argument and is pending validation. CPU success does not satisfy the two-camera image acceptance criterion or establish EGL, model precision, memory requirements or official-policy compatibility. The original upstream source remains unmodified. Python differs from the official policy evaluation reference (3.10.14), so the CPU environment is not claimed to reproduce the policy results.

## Read-only data audit

`scripts/audit_libero_data.py` uses h5py, NumPy, Pillow, protobuf and CRC32C with CUDA disabled. It reads both downloaded representations directly; it neither transforms nor filters examples.

```bash
.venv-libero/bin/python scripts/audit_libero_data.py \
  --original /path/to/libero-original \
  --rlds /path/to/oft-rlds \
  --source /path/to/pinned/LIBERO \
  --output /path/to/new-audit.json
```

Coverage is intentionally explicit:

- Every expected original task name is checked against the pinned source task map. All HDF5 episode schemas, numeric actions/states/rewards/dones, and first/last images for both cameras of every episode are read. Intermediate image pixels are not exhaustively decoded. Final demonstration reward labels are recorded, not treated as a fresh native success evaluation.
- All four RLDS suites have every TFRecord frame, length/payload CRC32C, Example numeric field, sequence length and first/last boundary checked. The first and last JPEG for both cameras of the first episode in each suite are decoded. This is a standard Example-wire parser, not a TensorFlow/TFDS training-loader integration test.
- Original files contain 128x128 RGB images; RLDS declares and stores 256x256 JPEG images. Identical underlying task names do not establish preprocessing or action-semantic equivalence between representations.
- Corrupted length CRC, corrupted payload CRC and truncated TFRecord fixtures were rejected by the actual reader; a valid fixture was accepted.

The full read audit completed successfully on 2026-10-29 (Asia/Shanghai). [Machine-readable results](../results/data_read_audit.json) record the exact script fingerprint and inspection scope. Original data: all 130 tasks, 6,500 demonstrations, 1,007,618 steps; all numeric arrays read, 26,000 first/last camera frames read, zero recorded errors. RLDS: all 96 TFRecord shards, 1,693 episodes and 273,465 steps; CRC and structural checks passed with zero errors; 16 JPEG endpoint samples decoded.

| Suite | Original episodes | Original steps | RLDS episodes | RLDS steps |
| --- | --- | --- | --- | --- |
| Spatial | 500 | 62,250 | 432 | 52,970 |
| Object | 500 | 74,507 | 454 | 66,984 |
| Goal | 500 | 63,728 | 428 | 52,042 |
| Long | 500 | 138,090 | 379 | 101,469 |
| LIBERO-90 | 4,500 | 669,043 | Not part of OFT four-suite input | Not applicable |

The processed input and raw upstream dataset have different episode coverage. No one-to-one episode correspondence or cause of exclusions is established by this audit. In particular, positive final reward labels in the original files are stored demonstration labels, not fresh simulator replays proving success. Neither all intermediate image pixels nor a TensorFlow/TFDS training loader have been validated. Source-download hashes previously passed for both full datasets; those checks are distinct from this read audit.
