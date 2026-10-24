# Feature map

| Journey | Driver | Proof |
|---|---|---|
| Environment | doctor | actual dependency versions and CUDA availability |
| [Sharing correctness](correctness.md) | drive | no-share parity, causal decode, alias storage, CKA/topology, gradient restrictions |
| [Pretrained distillation](pretrained.md) | pretrained | pinned real generation, changed consumer weights, held-out KL/NLL/agreement/cache bytes |
| [Trainer quality study](metrics.md) | tools/trainer_metrics.py | all 30 candidate replays passed, all gates failed; five frozen identity decisions each evaluated all 2,336 test blocks, zero saving |
| [Native HQQ cache quality](quantized-metrics.md) | tools/quantized_metrics.py | Separate identity/HQQ8/HQQ4 validation selection and unopened validation-tail confirmation; no result is claimed until its receipts pass |

## Sub-features
Calibration, cache sharing, distillation and memory-quality policy comparison.
## How to get to it (user POV)
Run `python -m kvshare experiment` or the documented verification driver.
## Driving it with CLI
Use the three base driver commands in the skill for doctor, CPU mechanism checks and pretrained distillation. The separate native HQQ comparison has its own three-action workflow in [quantized-metrics.md](quantized-metrics.md). Exit status must be zero and inspect JSON side effects.
## Gotchas
Pretrained and HQQ study modes require already-cached safetensors and exclusive CUDA. CPU tests use random tiny models only for deterministic mechanism checks, never pretrained quality claims.
