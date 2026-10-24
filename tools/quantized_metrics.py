"""Selection and confirmatory quality checks for native Transformers HQQ caches."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kvshare

DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATA_FILES = {
    "wikitext-2-raw-v1/train-00000-of-00001.parquet": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "wikitext-2-raw-v1/validation-00000-of-00001.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "wikitext-2-raw-v1/test-00000-of-00001.parquet": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}
GATES = {"whole_nll_increase": 0.02, "whole_teacher_kl": 0.02,
         "decode_nll_increase": 0.02, "decode_teacher_kl": 0.02}
POLICIES = {"identity": None, "hqq8": 8, "hqq4": 4}
POLICY_CONFIG = {"identity": {"cache": "DynamicCache"},
                 "hqq8": {"cache": "QuantizedCache", "backend": "hqq", "nbits": 8,
                          "axis_key": 1, "axis_value": 1, "q_group_size": 64, "residual_length": 32},
                 "hqq4": {"cache": "QuantizedCache", "backend": "hqq", "nbits": 4,
                          "axis_key": 1, "axis_value": 1, "q_group_size": 64, "residual_length": 32}}
SNAPSHOT_LENGTHS = (64, 95, 96, 97, 127, 128)
VALIDATION_RANGE = (0, 128)
CONFIRMATORY_RANGE = (128, 384)
BATCH_SIZE = 4
SCHEDULE = {"batch_size": BATCH_SIZE, "validation_batches": 32, "confirmatory_batches": 64,
            "prefill_tokens": 64, "single_token_updates": 64, "whole_nll_targets": 127,
            "whole_kl_positions": 128, "decode_nll_targets": 63, "decode_kl_positions": 64,
            "cache_snapshot_lengths": list(SNAPSHOT_LENGTHS)}
ACTION_TIMEOUTS = {"validate": 600, "confirmatory": 1200}
GPU_ALLOCATION_LIMIT = 16 * 1024 ** 3
DISK_RESERVE = 10 * 1024 ** 3


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resource_guard(started, output, device, action):
    import torch

    if time.monotonic() - started > ACTION_TIMEOUTS[action]:
        raise TimeoutError(f"{action} exceeded its {ACTION_TIMEOUTS[action]} second budget")
    if shutil.disk_usage(output).free < DISK_RESERVE:
        raise RuntimeError("preserve at least 10 GiB free disk")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if max(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()) > GPU_ALLOCATION_LIMIT:
            raise RuntimeError("GPU allocation exceeded 16 GiB")


def resource_receipt(device):
    import torch

    result = {"requested_device": device, "cuda_available": torch.cuda.is_available(),
              "process_peak_rss_bytes": None, "process_peak_rss_status": "not measured"}
    if device == "cuda" and torch.cuda.is_available():
        result.update(actual_cuda_device=torch.cuda.get_device_name(),
                      current_allocated_bytes=torch.cuda.memory_allocated(),
                      current_reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      total_device_bytes=torch.cuda.get_device_properties(0).total_memory)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_hashes():
    return {str(path.relative_to(ROOT)): file_hash(path) for path in
            (ROOT / "kvshare.py", ROOT / "tools" / "quantized_metrics.py")}


def model_identity(model_path):
    model_path = Path(model_path).resolve()
    if model_path.name != kvshare.REVISION:
        raise ValueError(f"require cached model snapshot {kvshare.REVISION}")
    files = {path.name: file_hash(path) for path in sorted(model_path.iterdir())
             if path.suffix == ".safetensors" or path.name in
             ("config.json", "tokenizer.json", "tokenizer_config.json")}
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("model snapshot has no safetensors")
    return {"revision": kvshare.REVISION, "files": files}


def load_validation(data_path, expected_model):
    from datasets import load_from_disk

    data_path = Path(data_path)
    provenance_path = data_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (provenance["dataset"] != "Salesforce/wikitext" or provenance["revision"] != DATA_REVISION
            or provenance["file_sha256"] != DATA_FILES or provenance["context"] != 128
            or provenance["model_identity"] != expected_model):
        raise ValueError("WikiText/model provenance does not match the frozen data")
    for relative, expected in provenance["prepared_file_sha256"]["validation"].items():
        if file_hash(data_path / relative) != expected:
            raise ValueError(f"prepared validation data changed: {relative}")
    if provenance["splits"]["validation"]["blocks"] < 384:
        raise ValueError("validation split does not contain the frozen confirmatory tail")
    dataset = load_from_disk(str(data_path / "validation"))
    if len(dataset) != provenance["splits"]["validation"]["blocks"]:
        raise ValueError("prepared validation block count changed")
    return dataset, file_hash(provenance_path), provenance


def _cache_factory(policy, config):
    from transformers import DynamicCache
    return DynamicCache() if policy == "identity" else kvshare.new_hqq_cache(config, POLICIES[policy])


def _forward(model, cache, ids, mask):
    result = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True,
                   return_dict=True)
    if result.past_key_values is not cache:
        raise RuntimeError("model did not update the supplied cache")
    return result.logits


def _cache_state(cache, expected_layers, quantized, batch_size):
    layers = cache.layers
    if len(layers) != expected_layers:
        raise RuntimeError(f"cache has {len(layers)} layers; expected {expected_layers}")
    quantized_layers = sum(getattr(layer, "_quantized_keys", None) is not None for layer in layers)
    if quantized and quantized_layers != expected_layers:
        raise RuntimeError("native HQQ state is missing from one or more cache layers")
    state = {
        "sequence_length": int(cache.get_seq_length()),
        "batch_size": batch_size,
        "quantized_layers": quantized_layers,
        "residual_tokens_by_layer": [int(layer.keys.shape[-2]) if getattr(layer, "keys", None) is not None
                                      and layer.keys.numel() else 0 for layer in layers],
        "persistent_bytes": kvshare.unique_cache_bytes(cache),
    }
    if state["persistent_bytes"] <= 0:
        raise RuntimeError("cache has no measurable persistent tensor storage")
    state["persistent_bytes_per_sequence_derived"] = state["persistent_bytes"] / batch_size
    return state


def _measure_pair(teacher_logits, candidate_logits, target):
    import torch
    from torch.nn import functional as F

    teacher_logits, candidate_logits = teacher_logits.float(), candidate_logits.float()
    finite = torch.isfinite(teacher_logits).all() & torch.isfinite(candidate_logits).all()
    teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    candidate_log_prob = F.log_softmax(candidate_logits, dim=-1)
    kl = (teacher_prob * (teacher_log_prob - candidate_log_prob)).sum(-1)
    return (F.cross_entropy(teacher_logits, target, reduction="none"),
            F.cross_entropy(candidate_logits, target, reduction="none"), kl,
            (teacher_logits.argmax(-1) == candidate_logits.argmax(-1)).long(), finite)


def evaluate_batch(model, tokens, policy):
    """Evaluate 1..4 independent blocks with one shared prefill/decode schedule."""
    import torch

    if tokens.ndim != 2 or tokens.shape[1] != 128 or not 1 <= tokens.shape[0] <= BATCH_SIZE:
        raise ValueError(f"expected a batch of 1..{BATCH_SIZE} prepared 128-token blocks")
    batch_size = tokens.shape[0]
    teacher_cache = _cache_factory("identity", model.config)
    candidate_cache = _cache_factory(policy, model.config)
    total_names = ("teacher_whole_nll", "candidate_whole_nll", "whole_kl", "teacher_decode_nll",
                   "candidate_decode_nll", "decode_kl", "seam_teacher_nll", "seam_candidate_nll")
    totals = {name: torch.zeros(batch_size, dtype=torch.float64, device=tokens.device) for name in total_names}
    counts = {name: torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
              for name in ("whole_agreement", "decode_agreement")}
    snapshots = {}
    finite_logits = torch.ones((), dtype=torch.bool, device=tokens.device)

    def add(logits_teacher, logits_candidate, position, target):
        nonlocal finite_logits
        values = _measure_pair(logits_teacher, logits_candidate, target)
        teacher_nll, candidate_nll, kl, agree, finite = values
        finite_logits &= finite
        totals["whole_kl"] += kl
        if position == 63:
            totals["seam_teacher_nll"] += teacher_nll
            totals["seam_candidate_nll"] += candidate_nll
        if position < 127:
            totals["teacher_whole_nll"] += teacher_nll
            totals["candidate_whole_nll"] += candidate_nll
        counts["whole_agreement"] += agree
        if position >= 64:
            totals["decode_kl"] += kl
            counts["decode_agreement"] += agree
            if position < 127:
                totals["teacher_decode_nll"] += teacher_nll
                totals["candidate_decode_nll"] += candidate_nll

    def snapshot(length):
        if length in SNAPSHOT_LENGTHS:
            layer_count = model.config.num_hidden_layers
            actual_teacher = _cache_state(teacher_cache, layer_count, False, batch_size)
            actual_candidate = _cache_state(candidate_cache, layer_count, policy != "identity", batch_size)
            if actual_teacher["sequence_length"] != length or actual_candidate["sequence_length"] != length:
                raise RuntimeError(f"cache length mismatch at scheduled boundary {length}")
            snapshots[str(length)] = {"teacher": actual_teacher, "candidate": actual_candidate}

    with torch.inference_mode():
        prefill_mask = torch.ones((batch_size, 64), dtype=torch.long, device=tokens.device)
        teacher_prefill = _forward(model, teacher_cache, tokens[:, :64], prefill_mask)
        candidate_prefill = _forward(model, candidate_cache, tokens[:, :64], prefill_mask)
        snapshot(64)
        for position in range(64):
            add(teacher_prefill[:, position], candidate_prefill[:, position], position,
                tokens[:, position + 1])
        del teacher_prefill, candidate_prefill
        for position in range(64, 128):
            mask = torch.ones((batch_size, position + 1), dtype=torch.long, device=tokens.device)
            token = tokens[:, position:position + 1]
            teacher_logits = _forward(model, teacher_cache, token, mask)[:, -1]
            candidate_logits = _forward(model, candidate_cache, token, mask)[:, -1]
            add(teacher_logits, candidate_logits, position,
                tokens[:, position + 1] if position < 127 else tokens[:, position])
            snapshot(position + 1)
            del teacher_logits, candidate_logits

    if not finite_logits.item():
        raise RuntimeError("non-finite logits")

    if set(snapshots) != {str(length) for length in SNAPSHOT_LENGTHS}:
        raise RuntimeError("cache schedule missed a required persistence boundary")

    totals = {name: values.tolist() for name, values in totals.items()}
    counts = {name: values.tolist() for name, values in counts.items()}
    rows = []
    for index in range(batch_size):
        teacher_nll = totals["teacher_whole_nll"][index] / 127
        candidate_nll = totals["candidate_whole_nll"][index] / 127
        decode_teacher_nll = totals["teacher_decode_nll"][index] / 63
        decode_candidate_nll = totals["candidate_decode_nll"][index] / 63
        row = {"batch_size": batch_size,
               "teacher_nll": teacher_nll,
               "candidate_nll": candidate_nll,
               "whole_nll_increase": candidate_nll - teacher_nll,
               "teacher_kl": totals["whole_kl"][index] / 128,
               "token_agreement": counts["whole_agreement"][index] / 128,
               "decode_teacher_nll": decode_teacher_nll,
               "decode_candidate_nll": decode_candidate_nll,
               "decode_nll_increase": decode_candidate_nll - decode_teacher_nll,
               "decode_teacher_kl": totals["decode_kl"][index] / 64,
               "decode_token_agreement": counts["decode_agreement"][index] / 64,
               "seam_target_nll_increase": totals["seam_candidate_nll"][index] - totals["seam_teacher_nll"][index],
               "whole_nll_targets": 127, "whole_kl_positions": 128,
               "decode_nll_targets": 63, "decode_kl_positions": 64,
               "prefill_cache_argument": True, "decode_cache_argument_calls": 64,
               "cache_snapshots": {length: {side: dict(state) for side, state in pair.items()}
                                   for length, pair in snapshots.items()}}
        if not all(math.isfinite(value) for key, value in row.items() if isinstance(value, float)):
            raise RuntimeError(f"non-finite block metrics at batch row {index}")
        rows.append(row)
    return rows


def evaluate_block(model, tokens, policy):
    """Single-example wrapper for tests and one-block diagnostics."""
    if tokens.shape != (1, 128):
        raise ValueError("expected one prepared 128-token block")
    return evaluate_batch(model, tokens, policy)[0]


def summarize(rows):
    if not rows or len({row["block_id"] for row in rows}) != len(rows):
        raise ValueError("empty metrics or duplicate block IDs")
    first_block = rows[0]["block_id"]
    if any(row["batch_size"] != BATCH_SIZE
           or row["batch_index"] != (row["block_id"] - first_block) // BATCH_SIZE
           for row in rows):
        raise ValueError("metrics rows do not match contiguous fixed-size batches")
    fields = ("teacher_nll", "candidate_nll", "teacher_kl", "token_agreement",
              "decode_teacher_nll", "decode_candidate_nll", "decode_teacher_kl", "decode_token_agreement")
    if any(not math.isfinite(row[key]) for row in rows for key in fields):
        raise ValueError("non-finite block metrics")
    whole_deltas = [row["candidate_nll"] - row["teacher_nll"] for row in rows]
    decode_deltas = [row["decode_candidate_nll"] - row["decode_teacher_nll"] for row in rows]
    mean = lambda values: statistics.fmean(values)
    result = {f"mean_{key}": mean([row[key] for row in rows]) for key in fields}
    result.update(whole_nll_increase=mean(whole_deltas), decode_nll_increase=mean(decode_deltas),
                  whole_nll_delta_stdev=statistics.stdev(whole_deltas) if len(rows) > 1 else 0.0,
                  decode_nll_delta_stdev=statistics.stdev(decode_deltas) if len(rows) > 1 else 0.0,
                  whole_nll_delta_min=min(whole_deltas), whole_nll_delta_max=max(whole_deltas),
                  decode_nll_delta_min=min(decode_deltas), decode_nll_delta_max=max(decode_deltas),
                  blocks=len(rows))
    for side, label in (("candidate", "persistent_bytes"), ("teacher", "teacher_persistent_bytes")):
        result[label] = {}
        result[f"{label}_per_sequence_derived"] = {}
        for length in ("64", "95", "96", "97", "127", "128"):
            states = [row["cache_snapshots"][length][side] for row in rows]
            if any(state.get("sequence_length") != int(length)
                   or state.get("batch_size") != row["batch_size"]
                   or state.get("persistent_bytes", 0) <= 0
                   or state.get("persistent_bytes_per_sequence_derived") !=
                      state["persistent_bytes"] / state["batch_size"]
                   for row, state in zip(rows, states)):
                raise ValueError(f"invalid {side} persistent storage receipt at length {length}")
            by_batch = {}
            for row, state in zip(rows, states):
                batch = row["batch_index"]
                comparable = (state["persistent_bytes"], state["persistent_bytes_per_sequence_derived"],
                              state.get("quantized_layers"), tuple(state.get("residual_tokens_by_layer", [])))
                if batch in by_batch and by_batch[batch] != comparable:
                    raise ValueError(f"inconsistent {side} cache storage within batch {batch}")
                by_batch[batch] = comparable
            result[label][length] = mean([state["persistent_bytes"] for state in states])
            result[f"{label}_per_sequence_derived"][length] = mean(
                [state["persistent_bytes_per_sequence_derived"] for state in states])
    result["batch_size"] = rows[0]["batch_size"]
    result["batches"] = len({row["batch_index"] for row in rows})
    result["persistent_bytes_scope"] = "actual total unique tensor storage for the whole batch"
    result["whole_gate_passed"] = (result["whole_nll_increase"] <= GATES["whole_nll_increase"]
                                    and result["mean_teacher_kl"] <= GATES["whole_teacher_kl"])
    result["decode_gate_passed"] = (result["decode_nll_increase"] <= GATES["decode_nll_increase"]
                                     and result["mean_decode_teacher_kl"] <= GATES["decode_teacher_kl"])
    result["passed"] = passes_gate({"whole_nll_increase": result["whole_nll_increase"],
        "whole_teacher_kl": result["mean_teacher_kl"], "decode_nll_increase": result["decode_nll_increase"],
        "decode_teacher_kl": result["mean_decode_teacher_kl"]})
    return result


def passes_gate(metrics):
    return all(math.isfinite(metrics[name]) and metrics[name] <= limit for name, limit in GATES.items())


def runtime_identity(model, device):
    import torch
    import transformers

    result = {"python": sys.version, "torch": torch.__version__,
              "transformers": transformers.__version__, "hqq": importlib.metadata.version("hqq"),
              "device": str(next(model.parameters()).device),
              "dtype": str(next(model.parameters()).dtype)}
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        result.update(cuda_build=torch.version.cuda, device_name=properties.name,
                      device_capability=list(torch.cuda.get_device_capability(0)),
                      total_device_bytes=properties.total_memory)
    else:
        result.update(cuda_build=torch.version.cuda, device_name=None,
                      device_capability=None, total_device_bytes=None)
    return result


def choose_policy(candidates):
    baseline_bytes = candidates["identity"]["teacher_persistent_bytes"]["128"]
    passing = [(candidates[name]["persistent_bytes"]["128"],
                candidates[name]["whole_nll_increase"], name)
               for name in ("hqq8", "hqq4")
               if candidates[name]["passed"] and candidates[name]["persistent_bytes"]["128"] < baseline_bytes]
    return min(passing)[2] if passing else "identity"


def _model_and_data(model_path, data_path, device):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    expected_model = model_identity(model_path)
    dataset, provenance_sha, provenance = load_validation(data_path, expected_model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
              use_safetensors=True, dtype=dtype, attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    return model, dataset, expected_model, provenance_sha, provenance


def run_validate(args):
    import torch
    import transformers

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"action": "validate", "status": "running", "command": sys.argv,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes(),
              "gates": GATES, "policy_config": POLICY_CONFIG, "schedule": SCHEDULE,
              "batch_size": BATCH_SIZE, "persistent_bytes_scope": "actual total unique tensor storage for the whole batch"}
    write_json(output / "result.json", result)
    try:
        resource_guard(started, output, args.device, "validate")
        model, dataset, identity, provenance_sha, provenance = _model_and_data(args.model, args.data, args.device)
        resource_guard(started, output, args.device, "validate")
        result.update(model_identity=identity, data_provenance_sha256=provenance_sha,
                      prepared_validation_sha256=provenance["prepared_file_sha256"]["validation"],
                      data_path=str(Path(args.data).resolve()), model_path=str(Path(args.model).resolve()),
                      block_range=list(VALIDATION_RANGE), device=str(next(model.parameters()).device),
                      dtype=str(next(model.parameters()).dtype), runtime=runtime_identity(model, args.device))
        active_policy = None
        for name, nbits in POLICIES.items():
            active_policy = name
            path = output / f"{name}-blocks.jsonl"
            rows = []
            with path.open("w", encoding="utf-8") as stream:
                for batch_index, batch_start in enumerate(range(VALIDATION_RANGE[0], VALIDATION_RANGE[1], BATCH_SIZE)):
                    resource_guard(started, output, args.device, "validate")
                    indices = range(batch_start, batch_start + BATCH_SIZE)
                    examples = [dataset[index] for index in indices]
                    if [example["block_id"] for example in examples] != list(indices):
                        raise ValueError("validation block IDs are not ordered")
                    tokens = torch.tensor([example["input_ids"] for example in examples],
                                          dtype=torch.long, device=args.device)
                    batch_rows = evaluate_batch(model, tokens, name)
                    for index, row in zip(indices, batch_rows):
                        row = {"block_id": index, "batch_index": batch_index, **row}
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.append(row)
                    stream.flush()
                    result.setdefault("coverage_blocks", {})[name] = len(rows)
                    result.setdefault("coverage_batches", {})[name] = batch_index + 1
                    resource_guard(started, output, args.device, "validate")
                    write_json(output / "result.json", result)
            metrics = summarize(rows)
            if metrics["batch_size"] != BATCH_SIZE or metrics["batches"] != SCHEDULE["validation_batches"]:
                raise RuntimeError("validation summary does not match the fixed batched schedule")
            metrics_path = output / f"{name}-metrics.json"
            write_json(metrics_path, metrics)
            result.setdefault("candidates", {})[name] = {
                "blocks": VALIDATION_RANGE[1] - VALIDATION_RANGE[0], "passed": metrics["passed"],
                "metrics_file": metrics_path.name,
                "metrics_sha256": file_hash(metrics_path), "blocks_file": path.name,
                "blocks_sha256": file_hash(path), "metrics": metrics}
            write_json(output / "result.json", result)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error),
                      failed_policy=locals().get("active_policy"))
        raise
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started, resources=resource_receipt(args.device))
        write_json(output / "result.json", result)


def read_validation_receipts(path):
    path = Path(path).resolve()
    result_path = path / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("action") != "validate":
        raise ValueError("validation result is incomplete")
    if (result["source_sha256"] != source_hashes() or result.get("policy_config") != POLICY_CONFIG
            or result.get("gates") != GATES or result.get("schedule") != SCHEDULE
            or result.get("batch_size") != BATCH_SIZE
            or result.get("coverage_blocks") != {policy: VALIDATION_RANGE[1] - VALIDATION_RANGE[0]
                                                   for policy in POLICIES}
            or result.get("coverage_batches") != {policy: SCHEDULE["validation_batches"]
                                                    for policy in POLICIES}):
        raise ValueError("validation source changed")
    receipts = {"result.json": file_hash(result_path)}
    for policy in POLICIES:
        candidate = result["candidates"][policy]
        if candidate["blocks"] != VALIDATION_RANGE[1] - VALIDATION_RANGE[0]:
            raise ValueError("validation candidate does not cover 128 blocks")
        for name, digest in ((candidate["metrics_file"], candidate["metrics_sha256"]),
                             (candidate["blocks_file"], candidate["blocks_sha256"])):
            if file_hash(path / name) != digest:
                raise ValueError("validation candidate receipt changed")
            receipts[name] = digest
        rows = [json.loads(line) for line in (path / candidate["blocks_file"]).read_text(encoding="utf-8").splitlines()]
        if [row["block_id"] for row in rows] != list(range(*VALIDATION_RANGE)):
            raise ValueError("validation candidate block coverage changed")
        if any(row["batch_size"] != BATCH_SIZE or row["batch_index"] != index // BATCH_SIZE
               for index, row in enumerate(rows)):
            raise ValueError("validation candidate batch coverage changed")
        reported_file = json.loads((path / candidate["metrics_file"]).read_text(encoding="utf-8"))
        if reported_file != candidate["metrics"]:
            raise ValueError("validation candidate metrics do not match their receipt")
        reported = candidate["metrics"]
        if (reported.get("batch_size") != BATCH_SIZE
                or reported.get("batches") != SCHEDULE["validation_batches"]
                or reported.get("blocks") != VALIDATION_RANGE[1] - VALIDATION_RANGE[0]):
            raise ValueError("validation metrics do not match the fixed batched workload")
        recomputed = summarize(rows)
        if recomputed != reported:
            raise ValueError("validation metrics do not recompute from block rows")
        if reported["passed"] != candidate["passed"]:
            raise ValueError("validation quality gate receipt is inconsistent")
    return result, receipts


def recompute_validation_choice(validation_path):
    result, receipts = read_validation_receipts(validation_path)
    if result.get("block_range") != list(VALIDATION_RANGE):
        raise ValueError("candidate selection requires the fixed validation range")
    validation_root = Path(validation_path).resolve()
    rows_by_policy = {
        name: [json.loads(line) for line in
               (validation_root / result["candidates"][name]["blocks_file"])
               .read_text(encoding="utf-8").splitlines()]
        for name in POLICIES
    }
    identity_rows = rows_by_policy["identity"]
    for policy in ("hqq8", "hqq4"):
        if any(row["teacher_nll"] != baseline["teacher_nll"]
               or row["cache_snapshots"]["128"]["teacher"] != baseline["cache_snapshots"]["128"]["teacher"]
               for row, baseline in zip(rows_by_policy[policy], identity_rows)):
            raise ValueError("candidates do not have the same teacher baseline")
    metrics = {name: result["candidates"][name]["metrics"] for name in POLICIES}
    return result, receipts, choose_policy(metrics)


def verify_frozen_selection(selection_path, expected_sha256):
    selection_path = Path(selection_path).resolve()
    if not expected_sha256 or file_hash(selection_path) != expected_sha256.lower():
        raise ValueError("frozen selection SHA-256 does not match the supplied digest")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("status") != "frozen" or selection.get("confirmatory_evaluated")
            or selection.get("source_sha256") != source_hashes()
            or selection.get("policy_config") != POLICY_CONFIG
            or selection.get("gates") != GATES
            or selection.get("schedule") != SCHEDULE
            or selection.get("validation_block_range") != list(VALIDATION_RANGE)
            or selection.get("confirmatory_block_range") != list(CONFIRMATORY_RANGE)):
        raise ValueError("selection is not a current frozen receipt for the fixed study")
    result, receipts, recomputed = recompute_validation_choice(selection["validation_path"])
    if (selection.get("validation_receipts") != receipts
            or selection.get("selected") != recomputed
            or selection.get("selected_nbits") != POLICIES[recomputed]
            or selection.get("model_identity") != result["model_identity"]
            or selection.get("runtime") != result.get("runtime")
            or selection.get("data_provenance_sha256") != result["data_provenance_sha256"]
            or selection.get("prepared_validation_sha256") != result["prepared_validation_sha256"]):
        raise ValueError("frozen selection differs from recomputed validation decision")
    return selection, result


def run_select(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection = {"status": "running", "action": "select",
                 "started_at_utc": datetime.now(timezone.utc).isoformat()}
    path = output / "selection.json"
    write_json(path, selection)
    try:
        result, receipts, selected = recompute_validation_choice(args.validation)
        selection.update(status="frozen", selected=selected, selected_nbits=POLICIES[selected],
                         selection_basis="lowest measured validation persistent bytes among passing HQQ candidates",
                         validation_path=str(Path(args.validation).resolve()), validation_receipts=receipts,
                         policy_config=POLICY_CONFIG,
                         schedule=SCHEDULE,
                         source_sha256=source_hashes(), model_identity=result["model_identity"],
                         runtime=result["runtime"],
                         data_provenance_sha256=result["data_provenance_sha256"],
                         prepared_validation_sha256=result["prepared_validation_sha256"], gates=GATES,
                         validation_block_range=list(VALIDATION_RANGE),
                         confirmatory_block_range=list(CONFIRMATORY_RANGE),
                         created_at_utc=datetime.now(timezone.utc).isoformat(), confirmatory_evaluated=False)
    except BaseException as error:
        selection.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(path, selection)


def _audit_unopened_tail(root, selection_path, validation_path=None):
    checked = []
    selection_path = Path(selection_path).resolve()
    validation_path = Path(validation_path).resolve() if validation_path else None
    paths = set((root / "evidence").rglob("*blocks.jsonl"))
    attempt_paths = set((root / "evidence").rglob("*attempts.jsonl"))
    if validation_path and validation_path.is_dir():
        paths.update(validation_path.rglob("*blocks.jsonl"))
        attempt_paths.update(validation_path.rglob("*attempts.jsonl"))
    for path in sorted(paths):
        if not path.is_file() or path.resolve() == selection_path:
            continue
        current = validation_path is not None and validation_path in path.resolve().parents
        if not current and "validation" not in str(path).lower() and "quantized" not in str(path).lower():
            continue
        ids = [json.loads(line)["block_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]
        overlap = sorted(set(ids) & set(range(*CONFIRMATORY_RANGE)))
        if overlap:
            raise ValueError(f"confirmatory validation tail already appears in receipt {path}: {overlap[:4]}")
        checked.append({"path": str(path.resolve()), "sha256": file_hash(path), "blocks": len(ids),
                        "max_block_id": max(ids) if ids else None})
    for path in sorted(attempt_paths):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            attempt = json.loads(line)
            overlap = sorted(set(attempt.get("block_ids", [])) & set(range(*CONFIRMATORY_RANGE)))
            if overlap:
                raise ValueError(f"confirmatory tail was already opened by attempt receipt {path}: {overlap[:4]}")
    return checked


def record_batch_attempt(path, batch_index, block_ids):
    """Durably record planned confirmation IDs immediately before their first model forward."""
    record = {"batch_index": batch_index, "block_ids": list(block_ids),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_confirmatory(args):
    import torch
    import transformers

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"action": "confirmatory", "status": "running", "command": sys.argv,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes(),
              "confirmatory_block_range": list(CONFIRMATORY_RANGE), "gates": GATES,
              "schedule": SCHEDULE, "batch_size": BATCH_SIZE,
              "policy_config": POLICY_CONFIG}
    write_json(output / "result.json", result)
    selection_path = Path(args.selection).resolve()
    try:
        resource_guard(started, output, args.device, "confirmatory")
        selection, _ = verify_frozen_selection(selection_path, args.selection_sha256)
        opened_receipts = _audit_unopened_tail(ROOT, selection_path, selection["validation_path"])
        result.update(selection_sha256=file_hash(selection_path), selection=selection,
                      opened_validation_receipts=opened_receipts)
        write_json(output / "result.json", result)
        model, dataset, identity, provenance_sha, provenance = _model_and_data(args.model, args.data, args.device)
        resource_guard(started, output, args.device, "confirmatory")
        if (identity != selection["model_identity"] or provenance_sha != selection["data_provenance_sha256"]
                or provenance["prepared_file_sha256"]["validation"] != selection["prepared_validation_sha256"]):
            raise ValueError("confirmatory model/data differs from frozen selection")
        if selection["selected"] not in POLICIES:
            raise ValueError("invalid selected policy")
        actual_runtime = runtime_identity(model, args.device)
        if actual_runtime != selection["runtime"]:
            raise ValueError("confirmatory runtime differs from frozen validation runtime")
        result.update(model_identity=identity, data_provenance_sha256=provenance_sha,
                      prepared_validation_sha256=provenance["prepared_file_sha256"]["validation"],
                      data_path=str(Path(args.data).resolve()), model_path=str(Path(args.model).resolve()),
                      device=str(next(model.parameters()).device), dtype=str(next(model.parameters()).dtype),
                      runtime=actual_runtime)
        write_json(output / "result.json", result)
        path = output / "confirmatory-blocks.jsonl"
        rows = []
        with path.open("w", encoding="utf-8") as stream:
            for batch_index, batch_start in enumerate(range(CONFIRMATORY_RANGE[0],
                                                            CONFIRMATORY_RANGE[1], BATCH_SIZE)):
                resource_guard(started, output, args.device, "confirmatory")
                indices = range(batch_start, batch_start + BATCH_SIZE)
                examples = [dataset[index] for index in indices]
                if [example["block_id"] for example in examples] != list(indices):
                    raise ValueError("confirmatory block IDs are not ordered")
                tokens = torch.tensor([example["input_ids"] for example in examples],
                                      dtype=torch.long, device=args.device)
                record_batch_attempt(output / "confirmatory-attempts.jsonl", batch_index, indices)
                batch_rows = evaluate_batch(model, tokens, selection["selected"])
                for index, row in zip(indices, batch_rows):
                    row = {"block_id": index, "batch_index": batch_index, **row}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                stream.flush()
                result["coverage_blocks"] = len(rows)
                result["coverage_batches"] = batch_index + 1
                resource_guard(started, output, args.device, "confirmatory")
                write_json(output / "result.json", result)
        if len(rows) != CONFIRMATORY_RANGE[1] - CONFIRMATORY_RANGE[0]:
            raise RuntimeError("confirmatory coverage did not match the frozen range")
        if result.get("coverage_batches") != SCHEDULE["confirmatory_batches"]:
            raise RuntimeError("confirmatory batch coverage did not match the fixed schedule")
        metrics = summarize(rows)
        if metrics["batch_size"] != BATCH_SIZE or metrics["batches"] != SCHEDULE["confirmatory_batches"]:
            raise RuntimeError("confirmatory summary does not match the fixed batched schedule")
        write_json(output / "confirmatory-metrics.json", metrics)
        result.update(status="complete", selected=selection["selected"], blocks=len(rows),
                      confirmatory_block_range=list(CONFIRMATORY_RANGE),
                      blocks_file=path.name, blocks_sha256=file_hash(path),
                      metrics_file="confirmatory-metrics.json", metrics_sha256=file_hash(output / "confirmatory-metrics.json"),
                      metrics=metrics)
    except BaseException as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started, resources=resource_receipt(args.device))
        write_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate")
    validate.add_argument("--model", required=True)
    validate.add_argument("--data", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    select = actions.add_parser("select")
    select.add_argument("--validation", required=True)
    select.add_argument("--output", required=True)
    confirmatory = actions.add_parser("confirmatory")
    confirmatory.add_argument("--model", required=True)
    confirmatory.add_argument("--data", required=True)
    confirmatory.add_argument("--selection", required=True)
    confirmatory.add_argument("--selection-sha256", required=True)
    confirmatory.add_argument("--output", required=True)
    confirmatory.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    {"validate": run_validate, "select": run_select, "confirmatory": run_confirmatory}[args.action](args)


if __name__ == "__main__":
    main()
