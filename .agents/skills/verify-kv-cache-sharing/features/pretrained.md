# Pretrained distillation

## Sub-features
Pinned public weights, iterative CKA, five held-out policies, actual generation, consumer training and checkpoint.

## How to get to it (user POV)
From repository root, run `python tools/verify.py doctor` then `python tools/verify.py pretrained`.

## Driving it with CLI
Require exit0. Inspect `evidence/pretrained.json`, actual generation strings, finite losses and positive weight delta.
Verify actual storage reduction and inspect every policy's quality metrics without assuming improvement.
Check `checkpoints/consumer.safetensors` exists. Preserve evidence after process exit.

## Gotchas
Requires the exact revision already cached and exclusive CUDA. Timeout or unavailable weights are failures.
The consumer window masks attention and retains producer cache storage. No quality/speed guarantee.
