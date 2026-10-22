# Trainer and WikiText study

`tools/trainer_metrics.py` is a thin adapter around Transformers Trainer and Accelerate. They own updates, seed handling, logging and checkpoint selection. Its loss calls the existing `kvshare.distill_loss`; evaluation calls the existing sharing hooks and `kvshare.metrics`. Only trainable consumer tensors are saved as safetensors. Product code and owned mechanism fixtures are unchanged.

The earlier two-update owned-fixture preflight and WikiText pilot remain compatibility receipts, not study samples. All **30 full candidates** now completed 256 updates, 128 validation blocks and exact checkpoint/validation replay: the original `evidence/wikitext-candidate-20261022-0701/` plus 29 cases under `evidence/wikitext-grid-20261022-0813/`. All failed both fixed quality gates. The frozen native selection at `evidence/wikitext-selection-20261022-0900/selection.json` chose identity/no-sharing for every seed before opening test. Its SHA-256 is `fde4941ae81f01a43a5c54ee0d14a7f1f9e1fe718329b92b45e0e3909dd90b0f`. All five decisions then evaluated all 2,336 official test blocks, preserving full token agreement and zero cache saving. Complete candidate/group/test metrics and hashes are curated in `docs/evidence/wikitext-metrics-20261022/summary.json`.

## Private framework environment

Project `.metrics-venv` exposes the existing pinned ML runtime without reinstalling Torch or Transformers. `requirements-metrics.txt` selects Accelerate 1.15.0 and Datasets 5.1.0, both Apache-2.0. `requirements-metrics.lock` contains Windows CPython 3.11 extra-wheel hashes; install with `--no-deps --require-hashes` after exposing the existing runtime, retain `constraints-metrics.txt`, and require `pip check`. [Trainer's extension API](https://huggingface.co/docs/transformers/main_classes/trainer) and the installed Transformers 5.20.2 source support the loss, prediction and consumer checkpoint overrides.

## Data and frozen workload

Explicit `prepare-data` downloads [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3), config `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. It checks each source Parquet SHA-256 and writes hashes of prepared Arrow files, tokenizer/model files and provenance. The [pinned card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) identifies CC-BY-SA 3.0/GFDL and Wikipedia-derived attribution. Do not commit or redistribute dataset text here.

Each official split is joined with two newlines and tokenized independently without special tokens. Split-local non-overlapping blocks contain 128 tokens; incomplete tails are excluded and counted. Blocks do not cross official split boundaries. Data preparation writes test blocks but performs no model forward or quality evaluation.

| Role | Frozen blocks, zero-based |
|---|---|
| Calibration only | Official train `[0, 4)` |
| Optimization only | Official train `[4, 260)` |
| Checkpoint/candidate selection only | Official validation `[0, 128)` |
| Final evaluation only | All complete official test blocks |

The full grid has seeds **11, 23, 37, 51, 71**, strategies **uniform and iterative CKA**, and **1, 2, 4** adjacent sharing pairs. Each of the 30 candidates starts from identical pinned Qwen3 weights and receives 256 Trainer updates, batch size 1, learning rate `2e-5`, AdamW, zero weight decay, linear schedule with no warmup, and gradient norm limit 1. The original temperature 2 / hidden-MSE weight 0.1 objective and consumer parameter restrictions are retained. The CKA map sees only calibration blocks. Uniform maps are `{2*i+1: 2*i}`.

Trainer validates and saves at updates **32, 128 and 256**, then reloads the checkpoint with lowest validation next-token NLL. All three consumer-only checkpoints and native JSON training logs are retained. Checkpoints omit optimizer/pickle state and support evaluation reload, not training resume.

## Validation gate and final-test boundary

For every validation block, evaluate the untouched frozen teacher and the candidate with the same BF16 eager attention and input IDs. Retain next-token NLL over 127 targets, teacher KL over 128 positions, teacher-forced per-position argmax token agreement, actual deduplicated cache bytes, independent cache bytes and block ID. Agreement does not evaluate generated rollouts or downstream task answers. Evaluation checks complete ordered block coverage and the exact aliasing byte reduction. Deterministic Torch algorithms are required; unsupported deterministic kernels fail rather than silently changing numerics.

Predeclared acceptance requires **mean paired NLL increase <= 0.02 nats/token** and **mean teacher KL <= 0.02 nats/token**. NLL 0.02 is an engineering goal of about 2.02% relative perplexity; KL 0.02 is a separate unvalidated drift budget. Neither is calibrated to downstream task accuracy or proves statistical equivalence, lossless sharing or task-quality preservation. Preserve each paired block difference, descriptive standard deviation and range; no significance claim is inferred from the threshold. The same untouched independent model supplies every paired baseline. Every candidate exceeded both limits: the smallest paired NLL increase was 0.2473799 and the smallest teacher KL 0.2625814.

The `select` action requires all **30** complete matched candidate receipts before any final test can run. For each seed, choose the passing checkpoint with lowest actual cache bytes, then lowest validation NLL, then strategy/pair count as a deterministic tie break. If none passes, choose the original **identity/no-sharing** model and report zero saving. Never choose a seed or retune a threshold using final test results.

`test-selected` is a separate action. It verifies all candidate receipt hashes, recomputes the frozen selection, checks the selected safetensors and model/data/source identities, then loads the official test split. All five seeds' decisions must already be frozen. It evaluates the selected model and paired teacher across every full test block without further training or selection. Preserve failures and report any test degradation even when validation passed.

## Commands and bounded pilot

Run only with the main agent's serialized resource grant, from the repository root using `.metrics-venv/Scripts/python.exe`:

```text
tools/trainer_metrics.py prepare-data --model <cached-snapshot> --output evidence/wikitext-data
tools/trainer_metrics.py pilot --model <cached-snapshot> --data evidence/wikitext-data --seed 11 --strategy cka --pairs 1 --output evidence/pilot
tools/trainer_metrics.py candidate --model <cached-snapshot> --data evidence/wikitext-data --seed 11 --strategy cka --pairs 1 --output evidence/seed11-cka1
tools/trainer_metrics.py select --candidates <all-30-result.json-paths> --output evidence/selection
tools/trainer_metrics.py test-selected --model <cached-snapshot> --data evidence/wikitext-data --seed 11 --selection evidence/selection/selection.json --output evidence/test-seed11
```

The executed pilot used eight updates, eight training blocks after calibration, eight validation blocks and checkpoints at 4/8; it cannot enter full selection. It passed within a 600-second owned-process-tree supervisor, using peak CUDA allocation 3,221,195,264 bytes and sampled tree RSS 2,140,266,496 bytes. It is a compatibility/correctness check, not convergence evidence. The completed full grid used **600-second external owned-tree caps**, stricter than the adapter's unchanged internal 7,200-second ceiling. All 29 new candidates completed within 68.000–106.469 seconds, with peak CUDA allocation 3,567,999,488 bytes. Each final evaluation likewise used an external 600-second cap and completed in 150.781–161.312 seconds, peak CUDA 1,799,724,032 bytes. External limits cover model loading, calibration and stuck kernels. The adapter checks time, a 16 GiB allocation cap and a 10 GiB free-disk reserve between model steps; save checks reserve before writing. Process exit releases CUDA; preserve receipts/checkpoints and stop only the task-owned process tree.

Final raw evidence is `evidence/wikitext-final-seed11-20261022-0900/` and `evidence/wikitext-final-remaining-20261022-0909/seed{23,37,51,71}/`, with adjacent supervisor/log/hash files. Every final result is identical because the selected identity weights and test corpus are the same; do not treat these as independent test-data samples. Mean NLL is 3.9101173804 versus baseline 3.9101173898; raw teacher KL is -2.378e-9. All 2,336 block IDs, finite outputs, full token agreement, equal cache bytes, source/data/model/selection identities and owned cleanup were checked. No adapter/backbone checkpoint was saved by final evaluation, and cached weight hashes remained unchanged.

The seed-11 model child exited 0, but an extra external audit incorrectly demanded bitwise equality between default-mean cross entropy and the product's unreduced-then-masked-mean formula. Its failure remains preserved. The five fixed-input CPU controls under `evidence/ce-reduction-control-20261022-0904/` reproduced scalar differences up to 9.537e-7 and signed self-KL roundoff with identical inputs. Later audits retain signed values and check finite outputs/token/cache/coverage rather than add a new scalar quality gate. This is consistent with numerical reduction effects; actual model logits were not retained for a bitwise-equality claim. No source, frozen gate or raw result was changed.

Receipts include `result.json`, native Trainer state, per-checkpoint validation block JSONL/metrics, selected-validation block JSONL/metrics, safe consumer checkpoints and final-test block receipts when explicitly run. Every output directory must be fresh. Five interleaved real decode comparisons and a separate profiler are still needed before any throughput claim; this quality adapter does not supply them. The earlier 7.14% cache saving degraded fixture NLL and remains a negative quality result.
