---
name: verify-kv-cache-sharing
description: Drive KV cache sharing's CPU correctness and real pinned pretrained experiment CLI.
---

## Launch
Use Python with torch 2.11.0 and transformers 5.20.2. Locally use
`C:/Users/steve/.codex/tools/coursework-ml/Scripts/python.exe` from repository root.
CLI processes are short lived; no persistent server or port is created.

## Doctor
`python tools/verify.py doctor` reports actual dependency and CUDA availability.

## Drive
`python tools/verify.py drive` drives CPU fixtures. `python tools/verify.py pretrained`
drives local-only pinned Qwen weights, calibration, generation, training, and held-out evaluation.
Both use subprocess timeouts; pretrained is limited to 600 seconds. Do not run concurrent GPU jobs.

## Evidence
`evidence/*-driver.json` retains command, stdout/stderr, exit code, elapsed time.
`evidence/pretrained.json` retains actual maps, metrics, generations and weight change.
Failures remain evidence; never present a timeout as a pass.

## Cleanup
The driver waits for its child; timeout kills that child. No persistent services.
CUDA allocations disappear with child exit. Preserve evidence; checkpoints are ignored local files.

## Helpers
The framework adapter, frozen Trainer grid, quality gate and final-test boundary are documented in [features/metrics.md](features/metrics.md). All 30 full candidates passed exact checkpoint/validation replay and failed the fixed quality gate. Selection chose identity/no-sharing for every seed; all five then completed the 2,336-block official test split with full token agreement and zero cache saving. Curated metrics/hashes are in `docs/evidence/wikitext-metrics-20261022/summary.json`. Preserve the seed-11 external scalar-audit failure and numerical-control receipts; no bitwise-logit or downstream quality claim follows.
`tools/verify.py` is the maintained driver. See `features/README.md` for mapped journeys.

## Native HQQ cache quality study

Use [`features/quantized-metrics.md`](features/quantized-metrics.md) and its three-action `tools/quantized_metrics.py` workflow for the independent native HQQ cache study. It uses the separate, hash-locked HQQ dependencies and the prepared WikiText validation split; it does not alter the Trainer study or access the official test split. Install only from the pinned project-local artifacts there, with `DISABLE_CUDA=1`.

The fixed sequence is `validate` on validation blocks `[0,128)`, `select` and freeze `selection.json`, then `confirmatory` on `[128,384)` with `--selection-sha256` computed from that frozen file. Both model actions use contiguous batches of four blocks and preserve per-block quality rows while reporting cache storage as actual whole-batch totals. The driver recomputes quality and byte summaries from those rows, freezes Python/Torch/Transformers/HQQ plus device/dtype/CUDA identity, and requires the same runtime before confirmation. It scans prior validation rows and durable confirmatory attempt receipts for held IDs. Each confirmatory batch's IDs are fsynced before its first model forward, so an interrupted batch prevents retrying the same tail as fresh. Require fresh output directories and preserve failures. Run model actions only under the main agent's exclusive GPU grant and owned-process-tree supervisor; the driver checks its 600/1,200-second action budgets, 10 GiB disk reserve and 16 GiB allocated-memory limit at batch boundaries, but cannot interrupt a stuck kernel itself. It reports CUDA allocated/reserved peaks and explicitly marks process RSS as unmeasured.

The retained one-block policy pilot under `evidence/hqq-policy-pilot-20261024/` used the pre-batching driver, so it is a preflight only and its metrics are not pooled with the four-example campaign. Before the full actions, run one four-example `evaluate_batch` pilot on the actual cached model; it is still not a study sample. Validation keeps HQQ8 and HQQ4 in its fixed candidate set and selects either only if measured batch-total persistent bytes are below identity and both whole-block and decode-phase NLL/KL gates pass. Confirmation reports the selected policy on the frozen validation tail and makes no throughput or downstream equivalence claim.
