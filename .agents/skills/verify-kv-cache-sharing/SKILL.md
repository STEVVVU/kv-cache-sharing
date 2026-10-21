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
`tools/verify.py` is the maintained driver. See `features/README.md` for mapped journeys.
