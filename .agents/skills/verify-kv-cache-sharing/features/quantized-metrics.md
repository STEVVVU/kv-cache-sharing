# Native HQQ cache quality study

This is a separate, inference-only comparison of Transformers' native `QuantizedCache(backend="hqq")` against an identity `DynamicCache`. It does not modify or extend the sharing/distillation Trainer study. It has no optimizer or seed grid and never loads the official test split.

## Fixed candidates and workload

The pinned Qwen3 snapshot and prepared WikiText-2 data must already exist locally. Data preparation and model loading are local-only. The driver verifies the prepared validation Arrow hashes, dataset provenance, model snapshot files, source hashes and Transformers/HQQ runtime versions before evaluating.

| Candidate | Cache |
|---|---|
| `identity` | Native `DynamicCache` control |
| `hqq8` | Transformers `QuantizedCache`, HQQ, 8 bits |
| `hqq4` | Transformers `QuantizedCache`, HQQ, 4 bits |

Both HQQ candidates use `axis_key=1`, `axis_value=1`, `q_group_size=64`, and `residual_length=32`. The driver groups four contiguous 128-token blocks per batch (32 batches for selection; 64 for confirmation). Each batch uses one 64-token prefill and then 64 single-token updates. Teacher and candidate use the same model object, dtype, eager attention, tokens and cache-fed schedule, with fresh caches for each side. Each block keeps its own metrics and block ID; quality statistics are never averaged across batch examples. No sharing hooks or training are used.

The validation selection range is official validation block IDs `[0,128)`. The held confirmatory range is `[128,384)`, exactly 256 distinct blocks. Selection and confirmatory evaluation are separate actions; the official test split is never available to this driver.

## Quality and storage measurements

The driver retains per-block paired differences, finite-logit checks, teacher-forced argmax agreement, and summaries for both phases:

- Whole block: 127 next-token targets and 128 teacher-KL positions. Prefill position 63 predicts token 64 and contributes only to whole-block NLL/KL; it is the seam before decode.
- Decode: 63 next-token targets from positions 64–126 and 64 KL positions from 64–127.
- Each NLL increase and mean teacher KL must be at most `0.02` for both whole and decode phases.

Persistent cache storage is measured from actual cache layer tensors and nested HQQ payload/metadata tensors, deduplicated by backing storage. The recorded `persistent_bytes` are the actual unique storage total for the whole four-example batch. Each per-block row labels this batch size and repeats its batch total; `persistent_bytes_per_sequence_derived` is a separately named total divided by four. Each block records observed cache state and bytes at sequence lengths 64, 95, 96, 97, 127 and 128; the driver does not assume which update flushes residual state. A selected HQQ candidate must pass all four quality gates and use fewer persistent bytes at length 128 than the identity teacher under the same batch schedule. Otherwise selection freezes `identity` and reports no saving.

Validation writes each completed batch's per-block rows immediately and keeps an error receipt on exceptions. `select` requires all three complete 128-block candidates with exactly 32 batches, checks their receipt hashes, recomputes all quality and cache-storage aggregates from the per-block rows, and freezes source/model/data/config/schedule/runtime and candidate-receipt hashes. Confirmatory replay recomputes the fixed aggregates from the same rows and rejects altered summaries even if receipt hashes are updated. Runtime identity includes Python, Torch, Transformers, HQQ, device/dtype, and CUDA device/build identity when applicable; confirmation must match it before the first model forward. Before confirmation, the driver verifies the selection digest and replays its decision, then scans prior validation rows and durable confirmatory attempt receipts for any IDs in `[128,384)`. Each batch's IDs are fsynced to an attempts receipt immediately before its first forward, so an interrupted first batch also taints those IDs. Any previously opened tail blocks retry as fresh confirmation. Output directories must be new; errors, attempt records and partial block receipts remain in place.

## Retained one-block pilot

The pre-batching actual-model pilot is preserved in `evidence/hqq-policy-pilot-20261024/result.json` with its source copies and hashes. On that single block, HQQ8 had decode NLL increase `0.00184797`, decode KL `0.00262305`, and length-128 cache bytes `7,798,784` versus `14,680,064` for identity. HQQ4 had decode NLL increase `2.25329807`, decode KL `3.08574362`, and cache bytes `4,128,768`. This diagnostic does not select a candidate: retain both candidates for the fixed validation grid, keep all gates unchanged, and do not pool or compare its metrics with the new four-example batched campaign. Repeat a one-batch `evaluate_batch` pilot on the actual cached model before full validation to confirm runtime, batch numerics and memory.

## Optional HQQ installation

The pinned ML runtime is already in `.metrics-venv`. HQQ is an optional runtime dependency and is imported only when this study constructs an HQQ cache. The project lock and local hash-checked artifacts avoid CUDA extension compilation and dependency resolution from the network:

```powershell
$python = '.metrics-venv/Scripts/python.exe'
$wheelhouse = 'evidence/hqq-setup-20261024'
$env:DISABLE_CUDA = '1'
& $python -m pip install --no-index --find-links $wheelhouse --no-deps --require-hashes -r "$wheelhouse/build-requirements.lock"
& $python -m pip install --no-index --find-links $wheelhouse --no-build-isolation --no-deps --require-hashes -r requirements-quantized.lock
& $python -m pip check
Remove-Item Env:DISABLE_CUDA
```

The first locked install supplies the local `wheel` build tool for the HQQ source archive. `DISABLE_CUDA=1` selects HQQ's ATen path and prevents its setup script from invoking CUDA extension installation. Preserve the source/hashes and install transcripts under `evidence/hqq-setup-20261024/`. Do not install a second Torch or Transformers build.

## Commands

Use `.metrics-venv/Scripts/python.exe` from the repository root. The data directory below is the already prepared, provenance-checked dataset. All three commands require fresh output directories. Only run `validate` and `confirmatory` with the main agent's serialized GPU/resource authorization and owned-process-tree timeout; do not overlap GPU jobs. Before the full validation action, the resource owner should run a one-block identity/HQQ8/HQQ4 pilot on the actual cached model as an operational check; it is not a selection sample or a quality result.

```powershell
$python = '.metrics-venv/Scripts/python.exe'
$model = '<local snapshot directory ending in c1899de289a04d12100db370d81485cdf75e47ca>'
$data = 'evidence/wikitext-data-20261022-0644'
$validation = 'evidence/quantized-validation-20261024'
$selectionDir = 'evidence/quantized-selection-20261024'
$selection = "$selectionDir/selection.json"
$confirmatory = 'evidence/quantized-confirmatory-20261024'

& $python tools/quantized_metrics.py validate --model $model --data $data --output $validation
& $python tools/quantized_metrics.py select --validation $validation --output $selectionDir
$selectionSha256 = (Get-FileHash $selection -Algorithm SHA256).Hash.ToLowerInvariant()
& $python tools/quantized_metrics.py confirmatory --model $model --data $data --selection $selection --selection-sha256 $selectionSha256 --output $confirmatory
```

The driver applies a 600-second validation and 1,200-second confirmatory budget, a 10 GiB free-disk reserve, and a 16 GiB CUDA allocated-memory cap at batch boundaries. Use an external owned-process-tree supervisor as well to stop a stuck kernel; the internal timer cannot interrupt a blocked GPU call. Receipts report the actual device, CUDA peak allocated/reserved bytes, runtime/source/model/data hashes, phase and batch coverage, and errors. Process peak RSS is explicitly marked unmeasured. Exit status and all JSON/JSONL receipts must be inspected; a partial/error receipt is not a pass.

This study establishes only the measured cache-byte and fixed teacher-forced metric results on these ranges. The thresholds are engineering gates, not a calibrated downstream task-quality or equivalence claim. It makes no throughput claim.
