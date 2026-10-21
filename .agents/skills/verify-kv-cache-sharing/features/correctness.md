# CPU correctness

## Sub-features
CKA/topology including full-capacity selection, actual aliasing including caller-supplied lazy caches, no-share parity, causal decoding, padding and gradient scopes.

## How to get to it (user POV)
From repository root, run `python tools/verify.py drive`.

## Driving it with CLI
Require exit0 and eight passing tests. Inspect `evidence/drive-driver.json` command and stderr.
The tests instantiate tiny actual Qwen3 modules and execute attention, caching and optimization.

## Gotchas
No pretrained quality claim follows from random tiny fixtures. They deliberately require no network or GPU.
