# OpenVLA-OFT paper reproduction scope

Scope expanded to all four OFT LIBERO evaluation suites on 2026-10-28. Tasks are executed sequentially after simulator and bounded policy validation, using one GPU at a time. No policy rollout, benchmark or training result has yet been obtained.

## Sources and target coverage

Paper: [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success, v2](https://arxiv.org/html/2502.19645v2). Code candidate: `e4287e94541f459edc4feabc4e181f537cd569a8`; see its [LIBERO guide](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LIBERO.md) and [evaluation implementation](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/run_libero_eval.py). Code compatibility remains unverified.

| Main suite | Tasks | Episodes per task and seed | Policy step limit | Matched checkpoint suffix |
| --- | --- | --- | --- | --- |
| Spatial | 10 | 50 | 220 | `libero-spatial` |
| Object | 10 | 50 | 280 | `libero-object` |
| Goal | 10 | 50 | 300 | `libero-goal` |
| Long (LIBERO-10) | 10 | 50 | 520 | `libero-10` |

Checkpoint IDs are `moojink/openvla-7b-oft-finetuned-` followed by the suffix above. Each checkpoint's exact revision, files and normalization statistics still need to be pinned before use. The main two-camera/proprioception configuration, rather than a single-camera ablation, is the initial target.

The official guide reports averaging three seeds. Full main coverage therefore means 4 suites x 10 tasks x 50 initial states x 3 seeds = 6,000 episodes. The code defaults to seed 7 but the guide does not identify all three publication seeds. Do not present a project-chosen seed list as the authors' seed list; freeze and disclose the actual list before formal evaluation.

The pinned code uses default task initial states indexed by episode, a ten-step settling period, two 256-pixel simulation cameras, center cropping for the policy, and eight actions executed before replanning. These are source-derived requirements, not verified interfaces. Normalization, camera orientation, gripper conversion, control semantics, model precision and termination still require runtime tracing. Infrastructure failures must be recorded separately: the upstream episode routine catches exceptions and returns an unsuccessful episode, so retaining only its success counter would hide error causes.

| Paper component | Reproduction status / requirement |
| --- | --- |
| Main four-suite OpenVLA-OFT success evaluation | Planned; 6,000-episode coverage target, prerequisites not passed |
| Combined four-suite policy (appendix) | Planned separately; published combined checkpoint exists; revisions and protocol pending |
| Inference latency / throughput | Planned; reproduce query protocol and report measured 4090 hardware, not A100-equivalent timing |
| Decoding, action representation, diffusion, input ablations | Coverage inventory created; each row needs matching weights or actual retraining; not reproduced by evaluating the main model |
| FiLM and pretrained-representation ablations | Require dedicated model/training conditions; not implemented |
| OFT training from the base model | Resource feasibility unresolved; no training started. Paper training uses eight 80GB A100/H100 GPUs; default upstream minimum exceeds the 24GB budget |
| ALOHA real-robot experiments / BridgeData extension | Outside current LIBERO simulation reproduction; no physical robot or validated implementation |

Downloading all data does not reproduce these experiments. Start with environment initialization, then one bounded official policy episode, one complete suite, and finally the other suites. Do not launch all variants simultaneously or claim training replication from inference-only results.

## Complete data acquisition

All byte sizes below come from pinned upstream file manifests, not estimated decompressed sizes.

| Dataset | Revision | Coverage | Bytes |
| --- | --- | --- | --- |
| [Original LIBERO](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) | `f13aa24a3da8c43c7225569f28c562979fa0e35a` | Spatial 10, Object 10, Goal 10, LIBERO-90, LIBERO-10: 130 tasks, 132 repository files | 100,442,945,547 |
| [OFT training RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds) | `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` | Four `*_no_noops` suites, 106 repository files | 10,230,693,345 |
| Existing IPEC LeRobot conversion | `e1a223d30b896c1613f270a2bfc63d382b3de7e1` | LIBERO-10 only; download hashes verified, decoder compatibility pending | 635,889,078 |

The original and RLDS copies are intentionally distinct: source demonstrations versus the paper's processed training input. The IPEC conversion does not replace either. LIBERO-90 is included for complete dataset coverage but is not an extra OFT main-result suite. Dataset access does not imply permission to redistribute or merge licenses; retain upstream notices and provenance separately.

Use `scripts/download_hf_dataset.py` on Linux with an upstream `revision?blobs=true` manifest, a fresh dataset directory and a state file. It locks each job, checks free space, resumes partial downloads, uses bounded worker counts and streaming hashes, and accepts each file only after upstream size and SHA256 (LFS) or Git blob SHA1 verification. State includes active partial-file bytes so transferred and fully verified bytes remain distinct. It does not create a duplicate Hugging Face blob cache. Runtime HDF5/RLDS parsing and episode coverage audits are required after download verification.

Both complete-data downloads finished on 2026-10-28. Every pinned-manifest file passed upstream size/hash verification; 132 original files and 106 RLDS files were also checked against their final on-disk sizes. This is download integrity, not data-read or policy validation. Detailed process IDs, live state paths, original deletion receipt and audit evidence are kept outside Git. The formal evaluation protocol remains unfrozen until checkpoint and runtime requirements are resolved.
