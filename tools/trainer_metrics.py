"""Trainer distillation and validation-only selection; final test is a separate action."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
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
DATASET = "Salesforce/wikitext"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATA_FILES = {
    "wikitext-2-raw-v1/train-00000-of-00001.parquet": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "wikitext-2-raw-v1/validation-00000-of-00001.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "wikitext-2-raw-v1/test-00000-of-00001.parquet": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}
STUDY = {
    "seeds": [11, 23, 37, 51, 71], "strategies": ["uniform", "cka"], "pairs": [1, 2, 4],
    "context": 128, "calibration_blocks": [0, 4], "training_blocks": [4, 260],
    "validation_blocks": [0, 128], "updates": 256, "checkpoints": [32, 128, 256],
    "learning_rate": 2e-5, "batch_size": 1, "weight_decay": 0.0,
    "optimizer": "adamw_torch", "lr_scheduler": "linear", "warmup_steps": 0,
    "max_grad_norm": 1.0, "distillation_temperature": 2.0, "hidden_mse_alpha": 0.1,
    "max_mean_paired_nll_increase": 0.02, "max_mean_teacher_kl": 0.02,
    "gate_units": "nats/token", "checkpoint_selection": "lowest validation NLL",
    "candidate_selection": "lowest cache bytes, then validation NLL, then strategy/pairs",
}
FIELDS = ("block_id", "next_token_nll", "baseline_nll", "teacher_kl", "token_agreement",
          "unique_cache_bytes", "baseline_cache_bytes")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def summarize(rows):
    """Equal-length blocks have equal 127-target NLL and 128-position KL weights."""
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("empty or non-finite evaluation")
    if len({row["block_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate evaluation block")
    deltas = [row["next_token_nll"] - row["baseline_nll"] for row in rows]
    return {
        **{key: statistics.fmean(row[key] for row in rows) for key in FIELDS[1:]},
        "mean_paired_nll_increase": statistics.fmean(deltas),
        "paired_nll_delta_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "paired_nll_delta_min": min(deltas), "paired_nll_delta_max": max(deltas),
        "blocks": len(rows),
    }


def passes_gate(metrics):
    return (all(math.isfinite(metrics[name]) for name in
                ("mean_paired_nll_increase", "teacher_kl"))
            and metrics["mean_paired_nll_increase"] <= STUDY["max_mean_paired_nll_increase"]
            and metrics["teacher_kl"] <= STUDY["max_mean_teacher_kl"])


def choose_candidate(candidates):
    """Require the entire frozen grid before selecting; identity is a valid outcome."""
    expected = {(strategy, pairs) for strategy in STUDY["strategies"] for pairs in STUDY["pairs"]}
    if len(candidates) != len(expected) or {(r["strategy"], r["pairs"]) for r in candidates} != expected:
        raise ValueError("selection requires all six distinct candidates for one seed")
    for row in candidates:
        if (row["study"] != STUDY or row["pilot_only"] or row["test_evaluated"]
                or not row["passed"] or row["global_step"] != STUDY["updates"]
                or row["validation"]["blocks"] != 128 or not row["exact_checkpoint_reload"]):
            raise ValueError("incomplete, changed or test-contaminated candidate")
    for key in ("seed", "source_sha256", "model_identity", "data_provenance_sha256"):
        if any(row[key] != candidates[0][key] for row in candidates):
            raise ValueError(f"unmatched candidate {key}")
    if candidates[0]["seed"] not in STUDY["seeds"]:
        raise ValueError("seed outside frozen study")
    for key in ("baseline_nll", "baseline_cache_bytes"):
        if any(row["validation"][key] != candidates[0]["validation"][key] for row in candidates):
            raise ValueError(f"unmatched independent validation {key}")
    eligible = [row for row in candidates if passes_gate(row["validation"])]
    return min(eligible, key=lambda r: (r["validation"]["unique_cache_bytes"],
               r["validation"]["next_token_nll"], r["strategy"], r["pairs"])) if eligible else None


def choose_study(candidates):
    if len(candidates) != 30 or {r["seed"] for r in candidates} != set(STUDY["seeds"]):
        raise ValueError("freeze all five seeds' six-candidate grids before final test")
    for key in ("source_sha256", "model_identity", "data_provenance_sha256"):
        if any(row[key] != candidates[0][key] for row in candidates):
            raise ValueError(f"unmatched study {key}")
    return {str(seed): choose_candidate([r for r in candidates if r["seed"] == seed]) for seed in STUDY["seeds"]}


def load_selection(path, seed=11):
    selected = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = []
    for receipt in selected["candidate_receipts"]:
        if file_hash(receipt["path"]) != receipt["sha256"]:
            raise ValueError("candidate receipt changed after selection")
        candidates.append(json.loads(Path(receipt["path"]).read_text(encoding="utf-8")))
    choices = choose_study(candidates)
    if selected["selected"] != choices or selected["study"] != STUDY or selected["test_evaluated"]:
        raise ValueError("selection does not match frozen validation decision")
    choice = choices[str(seed)]
    if choice and file_hash(choice["best_checkpoint_file"]) != choice["best_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    return choice, next(row for row in candidates if row["seed"] == seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare-data", "pilot", "candidate", "select", "test-selected"))
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, choices=STUDY["seeds"], default=11)
    parser.add_argument("--strategy", choices=STUDY["strategies"], default="uniform")
    parser.add_argument("--pairs", type=int, choices=STUDY["pairs"], default=1)
    parser.add_argument("--candidates", nargs="+", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.action == "select":
        if not args.candidates:
            parser.error("select requires --candidates <all 30 result.json paths>")
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidates]
        choices = choose_study(rows)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "selection.json", {
            "study": STUDY, "selected": choices,
            "outcome_by_seed": {seed: "sharing passed validation tolerance" if choice else
                                "identity/no-sharing fallback" for seed, choice in choices.items()},
            "candidate_receipts": [{"path": str(p.resolve()), "sha256": file_hash(p)} for p in args.candidates],
            "test_evaluated": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return
    if not args.model or (args.action in ("pilot", "candidate", "test-selected") and not args.data):
        parser.error("--model is required; study actions also require --data")
    if args.action == "test-selected" and not args.selection:
        parser.error("test-selected requires --selection")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = 600 if args.action in ("preflight", "pilot") else 7200
    if shutil.disk_usage(args.output).free < 10 * 1024 ** 3:
        raise RuntimeError("preserve at least 10 GiB free disk")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.nn import functional as F
    from datasets import Dataset, load_dataset, load_from_disk
    from safetensors.torch import load_file, save_file
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments, default_data_collator, set_seed)
    import kvshare as kv

    torch.set_num_threads(2)
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model_path = Path(args.model).resolve()
    if model_path.name != kv.REVISION:
        raise ValueError(f"require cached model snapshot {kv.REVISION}")
    model_identity = {"revision": kv.REVISION, "files": {p.name: file_hash(p) for p in
                      sorted(model_path.iterdir()) if p.suffix == ".safetensors" or
                      p.name in ("config.json", "tokenizer.json", "tokenizer_config.json")}}
    if not any(name.endswith(".safetensors") for name in model_identity["files"]):
        raise ValueError("missing local safetensors")
    if args.action == "prepare-data":
        from huggingface_hub import hf_hub_download
        for filename, expected in DATA_FILES.items():
            downloaded = Path(hf_hub_download(DATASET, filename, repo_type="dataset", revision=DATA_REVISION,
                                             cache_dir=str(args.output / "cache")))
            if file_hash(downloaded) != expected:
                raise ValueError(f"source hash mismatch: {filename}")
        raw = load_dataset(DATASET, "wikitext-2-raw-v1", revision=DATA_REVISION, cache_dir=str(args.output / "cache"))
        provenance = {"dataset": DATASET, "revision": DATA_REVISION, "file_sha256": DATA_FILES,
                      "license": ["CC-BY-SA-3.0", "GFDL"], "model_identity": model_identity,
                      "context": 128, "card": f"https://huggingface.co/datasets/{DATASET}/blob/{DATA_REVISION}/README.md",
                      "splits": {}, "prepared_file_sha256": {}}
        for name in ("train", "validation", "test"):
            ids = tokenizer("\n\n".join(raw[name]["text"]), add_special_tokens=False)["input_ids"]
            blocks = [{"block_id": i // 128, "input_ids": ids[i:i + 128], "attention_mask": [1] * 128,
                       "labels": ids[i:i + 128]} for i in range(0, len(ids) - 127, 128)]
            Dataset.from_list(blocks).save_to_disk(str(args.output / name))
            provenance["splits"][name] = {"blocks": len(blocks), "tokens": len(ids), "dropped_tail_tokens": len(ids) % 128}
            provenance["prepared_file_sha256"][name] = {str(p.relative_to(args.output)): file_hash(p)
                for p in sorted((args.output / name).rglob("*")) if p.is_file()}
        write_json(args.output / "provenance.json", provenance)
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    source_sha = {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "kvshare.py", Path(__file__))}
    provenance = None
    if args.action != "preflight":
        provenance = json.loads((args.data / "provenance.json").read_text(encoding="utf-8"))
        if (provenance["revision"] != DATA_REVISION or provenance["file_sha256"] != DATA_FILES
                or provenance["model_identity"] != model_identity or provenance["context"] != 128):
            raise ValueError("dataset/model provenance does not match frozen study")

    def split(name):
        # Selection/training callers cannot accidentally load the final test split.
        if name == "test" and args.action != "test-selected":
            raise ValueError("final test is unavailable during selection")
        for relative, expected in provenance["prepared_file_sha256"][name].items():
            if file_hash(args.data / relative) != expected:
                raise ValueError(f"prepared split changed: {relative}")
        return load_from_disk(str(args.data / name))

    chosen = None
    if args.action == "test-selected":
        chosen, matched = load_selection(args.selection, args.seed)
        if (matched["source_sha256"] != source_sha or matched["model_identity"] != model_identity
                or matched["data_provenance_sha256"] != file_hash(args.data / "provenance.json")):
            raise ValueError("final evaluation must use the selected code, model and data")
        args.seed = matched["seed"]
        args.strategy = chosen["strategy"] if chosen else "identity"
        args.pairs = chosen["pairs"] if chosen else 0
        set_seed(args.seed)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    teacher.requires_grad_(False)
    student = teacher if args.action == "test-selected" and chosen is None else copy.deepcopy(teacher)
    mapping, history = {}, []
    if args.action == "preflight":
        fixture = json.loads((ROOT / "fixtures.json").read_text(encoding="utf-8"))
        def encode(text, index):
            row = tokenizer(text, truncation=True, max_length=48)
            return {**row, "labels": row["input_ids"].copy(), "block_id": index}
        training = Dataset.from_list([encode(text, i) for i, text in enumerate(fixture["train"])])
        validation = Dataset.from_list([encode(text, i) for i, text in enumerate(fixture["calibration"])])
        mapping = {1: 0}
    elif args.action == "test-selected":
        mapping = {int(c): p for c, p in chosen["mapping"].items()} if chosen else {}
        validation = split("test")
        training = None
    else:
        train = split("train")
        training = train.select(range(4, 12 if args.action == "pilot" else 260))
        validation = split("validation").select(range(8 if args.action == "pilot" else 128))
        if args.strategy == "uniform":
            mapping = {2 * i + 1: 2 * i for i in range(args.pairs)}
        else:
            calibration = default_data_collator([train[i] for i in range(4)])
            inputs = {k: v.to("cuda") for k, v in calibration.items() if k in ("input_ids", "attention_mask")}
            mapping, history = kv.calibrate(student, inputs, args.pairs)
            del inputs, calibration
    route = kv.Sharing(student, mapping) if mapping else None
    parameters = kv.trainable_consumers(student, mapping) if mapping else []
    before = parameters[0].detach().float().clone() if parameters else None
    hidden_layer = min(max(mapping, default=0) + 1, len(student.model.layers) - 1)
    checkpoints = [1, 2] if args.action == "preflight" else [4, 8] if args.action == "pilot" else STUDY["checkpoints"]
    report = {"passed": False, "preflight_only": args.action == "preflight", "pilot_only": args.action == "pilot",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "study": STUDY,
              "model_identity": model_identity, "source_sha256": source_sha, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "seed": args.seed, "strategy": args.strategy, "pairs": args.pairs,
              "mapping": mapping, "calibration": history, "hidden_layer": hidden_layer, "test_evaluated": False,
              "data_provenance_sha256": file_hash(args.data / "provenance.json") if provenance else None,
              "framework": "Transformers Trainer+Accelerate", "deterministic_algorithms": True,
              "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory}

    def check_limits():
        if (time.monotonic() - started > deadline or shutil.disk_usage(args.output).free < 10 * 1024 ** 3
                or torch.cuda.max_memory_allocated() > 16 * 1024 ** 3):
            raise RuntimeError("run resource bound exceeded; coverage was not reduced")

    def load_consumers(path):
        saved = load_file(str(path), device="cuda")
        expected = {name: parameter for name, parameter in student.named_parameters() if parameter.requires_grad}
        if saved.keys() != expected.keys():
            raise ValueError("checkpoint must contain exactly this map's consumer parameters")
        with torch.no_grad():
            for name, parameter in expected.items():
                if saved[name].shape != parameter.shape or saved[name].dtype != parameter.dtype:
                    raise ValueError(f"checkpoint shape/dtype mismatch: {name}")
                parameter.copy_(saved[name])
        if not all(torch.equal(parameter, saved[name]) for name, parameter in expected.items()):
            raise RuntimeError("checkpoint reload was not exact")

    class CheckpointsAndLimits(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            check_limits()
        def on_step_end(self, args, state, control, **kwargs):
            control.should_evaluate = control.should_save = state.global_step in checkpoints
            return control

    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            check_limits()
            features = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                target = teacher(**features, output_hidden_states=True, use_cache=False)
            prediction = model(**features, output_hidden_states=True, use_cache=False)
            loss = kv.distill_loss(prediction, target, temperature=STUDY["distillation_temperature"],
                                   alpha=STUDY["hidden_mse_alpha"], hidden_layer=hidden_layer,
                                   mask=features["attention_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite distillation loss")
            return (loss, prediction) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            check_limits()
            inputs = self._prepare_inputs(inputs)
            features = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                target = teacher(**features, use_cache=True)
                baseline_nll = F.cross_entropy(target.logits[:, :-1].float().reshape(-1, target.logits.shape[-1]),
                                               features["input_ids"][:, 1:].reshape(-1))
                result = kv.metrics(model, features, target.logits)
                baseline_bytes = kv.unique_cache_bytes(target.past_key_values)
                if result["unique_cache_bytes"] * len(teacher.model.layers) != baseline_bytes * (len(teacher.model.layers) - len(mapping)):
                    raise RuntimeError("actual cache bytes do not match map aliasing")
                values = [inputs["block_id"].item(), result["next_token_nll"], float(baseline_nll),
                          result["teacher_kl"], result["token_agreement"], result["unique_cache_bytes"], baseline_bytes]
                prediction = torch.tensor([values], device=features["input_ids"].device, dtype=torch.float64)
            return None, prediction, inputs["block_id"][:, None]

        def save_model(self, output_dir=None, _internal_call=False):
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            tensors = {name: p.detach().contiguous().cpu() for name, p in self.model.named_parameters() if p.requires_grad}
            needed = sum(p.numel() * p.element_size() for p in tensors.values()) + 1024 ** 2
            if shutil.disk_usage(destination).free - needed < 10 * 1024 ** 3:
                raise RuntimeError("checkpoint would cross 10 GiB disk reserve")
            save_file(tensors, str(destination / "consumers.safetensors"))

        def _load_best_model(self):
            load_consumers(Path(self.state.best_model_checkpoint) / "consumers.safetensors")

    stage = None
    def compute_metrics(prediction):
        rows = [dict(zip(FIELDS, values.tolist())) for values in prediction.predictions]
        expected_ids = list(validation["block_id"])
        if [int(row["block_id"]) for row in rows] != expected_ids:
            raise ValueError("evaluation did not cover every requested block in order")
        metrics = summarize(rows)
        name = stage or f"validation-step-{trainer.state.global_step}"
        with (args.output / f"{name}-blocks.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(args.output / f"{name}-metrics.json", metrics)
        return metrics

    settings = TrainingArguments(output_dir=str(args.output / "checkpoints"), max_steps=checkpoints[-1],
               per_device_train_batch_size=1, per_device_eval_batch_size=1, learning_rate=STUDY["learning_rate"],
               weight_decay=STUDY["weight_decay"], seed=args.seed, data_seed=args.seed, bf16=True,
               optim=STUDY["optimizer"], lr_scheduler_type=STUDY["lr_scheduler"], warmup_steps=0, max_grad_norm=1.0,
               eval_strategy="steps", eval_steps=checkpoints[0], save_strategy="steps", save_steps=checkpoints[0],
               logging_steps=1, load_best_model_at_end=True, metric_for_best_model="eval_next_token_nll",
               greater_is_better=False, save_only_model=True, save_total_limit=3, prediction_loss_only=False,
               remove_unused_columns=False, report_to="none", dataloader_num_workers=0, label_names=["labels"])
    trainer = DistillationTrainer(model=student, args=settings, train_dataset=training, eval_dataset=validation,
                                 data_collator=default_data_collator, compute_metrics=compute_metrics,
                                 callbacks=[CheckpointsAndLimits()])
    try:
        if args.action == "test-selected":
            if chosen:
                load_consumers(chosen["best_checkpoint_file"])
            student.eval()
            stage = "final-test"
            trainer.predict(validation, metric_key_prefix="test")
            report.update(passed=True, test_evaluated=True, selection_sha256=file_hash(args.selection),
                          selected_policy="sharing" if chosen else "identity/no-sharing",
                          final_test=json.loads((args.output / "final-test-metrics.json").read_text()))
        else:
            stage = "untrained-validation"
            trainer.predict(validation, metric_key_prefix="initial")
            stage = None
            result = trainer.train()
            if trainer.state.global_step != checkpoints[-1] or not trainer.state.best_model_checkpoint:
                raise RuntimeError("Trainer did not complete the frozen update/checkpoint budget")
            checkpoint = (Path(trainer.state.best_model_checkpoint) / "consumers.safetensors").resolve()
            load_consumers(checkpoint)
            if any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("frozen teacher received gradients")
            delta = float((parameters[0].detach().float() - before).norm())
            if delta <= 0:
                raise RuntimeError("selected checkpoint did not change a tracked consumer parameter")
            stage = "selected-validation"
            trainer.predict(validation, metric_key_prefix="validation")
            measured = json.loads((args.output / "selected-validation-metrics.json").read_text())
            best_step = int(checkpoint.parent.name.rsplit("-", 1)[1])
            original = json.loads((args.output / f"validation-step-{best_step}-metrics.json").read_text())
            if measured != original:
                raise RuntimeError("reloaded checkpoint did not reproduce deterministic validation metrics")
            report.update(passed=True, global_step=trainer.state.global_step, trainer_metrics=result.metrics,
                          best_checkpoint_file=str(checkpoint), best_checkpoint_sha256=file_hash(checkpoint),
                          validation=measured, validation_gate_passed=passes_gate(measured),
                          parameter_delta=delta, teacher_frozen=True,
                          trainable_parameters=sum(p.numel() for p in parameters), checkpoint_bytes=checkpoint.stat().st_size,
                          exact_checkpoint_reload=True, exact_validation_replay=True, no_backbone_weights_saved=True)
        trainer.save_state()
    finally:
        if route:
            route.close()
        report.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
