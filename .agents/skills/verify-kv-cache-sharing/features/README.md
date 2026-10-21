# Feature map

| Journey | Driver | Proof |
|---|---|---|
| Environment | doctor | actual dependency versions and CUDA availability |
| [Sharing correctness](correctness.md) | drive | no-share parity, causal decode, alias storage, CKA/topology, gradient restrictions |
| [Pretrained distillation](pretrained.md) | pretrained | pinned real generation, changed consumer weights, held-out KL/NLL/agreement/cache bytes |

## Sub-features
Calibration, cache sharing, distillation and memory-quality policy comparison.
## How to get to it (user POV)
Run `python -m kvshare experiment` or the documented verification driver.
## Driving it with CLI
Use the three driver commands in the skill. Exit code must be zero and inspect JSON side effects.
## Gotchas
Pretrained mode requires already-cached safetensors and exclusive CUDA. CPU tests use random tiny models only for deterministic mechanism checks, never quality claims.
