import importlib.util
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import time
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import kvshare

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "quantized_metrics.py"
spec = importlib.util.spec_from_file_location("quantized_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
CHECKER = Path(__file__).resolve().parents[1] / "tools" / "check_hqq_receipts.py"
checker_spec = importlib.util.spec_from_file_location("check_hqq_receipts", CHECKER)
checker = importlib.util.module_from_spec(checker_spec)
checker_spec.loader.exec_module(checker)


class QuantizedMetricsTests(unittest.TestCase):
    def test_publication_relocates_with_lf_sources_and_no_original_evidence_access(self):
        project = SCRIPT.parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            relocated = Path(temporary)
            for name in ("kvshare.py", "tools/quantized_metrics.py", "tools/check_hqq_receipts.py"):
                target = relocated / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((project / name).read_bytes().replace(b"\r\n", b"\n"))
            shutil.copytree(project / "docs/evidence/hqq-cache-20261024", relocated / "bundle")
            code = """
import sys
from pathlib import Path
original = sys.argv[1].lower().replace('\\\\', '/')
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0]).lower().replace('\\\\', '/')
        if original in path and '/evidence/' in path:
            raise RuntimeError('original evidence is unavailable')
sys.addaudithook(audit)
sys.path.insert(0, str(Path.cwd() / 'tools'))
import check_hqq_receipts
assert check_hqq_receipts.check_receipts('bundle')['confirmatory_blocks'] == 256
"""
            completed = subprocess.run([sys.executable, "-c", code, str(project)], cwd=relocated,
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in checker.SOURCE_SHA256:
                path = relocated / name.replace("\\", "/")
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            checker.check_source_binding(relocated)
            (relocated / "kvshare.py").write_bytes(b"changed source\n")
            with self.assertRaisesRegex(ValueError, "source binding"):
                checker.check_source_binding(relocated)

    def test_publication_rejects_rehashed_raw_identity_and_selection_tampering(self):
        evidence = SCRIPT.parents[1] / "docs/evidence/hqq-cache-20261024"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            shutil.copytree(evidence, root)
            path = root / "validation/identity-metrics.json"
            path.write_bytes(path.read_bytes() + b"\n")
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"]["validation/identity-metrics.json"]["canonical_sha256"] = checker.digest(path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "raw receipt identity"):
                checker.check_receipts(root)
            shutil.rmtree(root)
            shutil.copytree(evidence, root)
            path = root / "selection/selection.json"
            selection = json.loads(path.read_text())
            selection["validation_path"] = "relocated/fabricated"
            path.write_bytes(json.dumps(selection).encode())
            manifest = json.loads((root / "manifest.json").read_text())
            entry = manifest["files"]["selection/selection.json"]
            entry["canonical_sha256"] = checker.digest(path)
            entry["raw_sha256"] = hashlib.sha256(path.read_bytes().replace(b"\n", b"\r\n")).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "raw receipt hash manifest"):
                checker.check_receipts(root)
            raw = Path(temporary) / "raw"
            for relative in manifest["files"]:
                target = raw / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((evidence / relative).read_bytes().replace(b"\n", b"\r\n"))
            path = raw / "confirmatory/result.json"
            result = json.loads(path.read_text())
            result["selection"]["validation_path"] = "fabricated"
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "confirmatory frozen selection"):
                checker.check_raw_receipts(raw)

    def test_published_hqq_receipts_recompute_without_loading_hqq(self):
        evidence = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "hqq-cache-20261024"
        self.assertEqual(checker.check_receipts(evidence), {
            "validation_blocks_per_candidate": 128,
            "selected": "hqq8",
            "confirmatory_blocks": 256,
            "confirmatory_batches": 64,
        })

    def test_unique_cache_bytes_counts_explicit_quantized_payload_and_metadata_once(self):
        base = torch.empty(32, dtype=torch.uint8)
        packed = torch.empty(16, dtype=torch.uint8)
        scale = torch.empty(2, dtype=torch.float32)
        zero = torch.empty(2, dtype=torch.float32)
        beam = torch.empty(3, dtype=torch.long)
        ignored = torch.empty(99, dtype=torch.uint8)
        layer = SimpleNamespace(keys=base, values=base.view(4, 8),
                                _quantized_keys=(packed, {"scale": scale}),
                                _quantized_values={"zero": zero}, _pending_beam_idx=beam,
                                unrelated=ignored)
        cache = SimpleNamespace(layers=[layer])
        expected = base.untyped_storage().nbytes() + packed.untyped_storage().nbytes()
        expected += scale.untyped_storage().nbytes() + zero.untyped_storage().nbytes()
        expected += beam.untyped_storage().nbytes()
        self.assertEqual(kvshare.unique_cache_bytes(cache), expected)

    @unittest.skipUnless(importlib.util.find_spec("hqq"), "optional HQQ backend is not installed")
    def test_hqq_factory_uses_fixed_native_policy_and_rejects_unregistered_bits(self):
        config = Qwen3Config(num_hidden_layers=2)
        cache = kvshare.new_hqq_cache(config, 8)
        self.assertEqual(cache.layers[0].nbits, 8)
        self.assertEqual(cache.layers[0].axis_key, 1)
        self.assertEqual(cache.layers[0].axis_value, 1)
        self.assertEqual(cache.layers[0].q_group_size, 64)
        self.assertEqual(cache.layers[0].residual_length, 32)
        with self.assertRaisesRegex(ValueError, "fixed 4-bit and 8-bit"):
            kvshare.new_hqq_cache(config, 3)

    @unittest.skipUnless(importlib.util.find_spec("hqq"), "optional HQQ backend is not installed")
    def test_real_cpu_hqq_decode_schedule_exercises_cache_and_phase_counts(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        config = Qwen3Config(vocab_size=32, hidden_size=128, intermediate_size=160,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=64)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        row = metrics.evaluate_block(model, torch.randint(0, 32, (1, 128)), "hqq8")
        self.assertEqual(row["whole_nll_targets"], 127)
        self.assertEqual(row["whole_kl_positions"], 128)
        self.assertEqual(row["decode_nll_targets"], 63)
        self.assertEqual(row["decode_kl_positions"], 64)
        self.assertTrue(row["prefill_cache_argument"])
        self.assertEqual(row["decode_cache_argument_calls"], 64)
        self.assertEqual(set(row["cache_snapshots"]), {"64", "95", "96", "97", "127", "128"})
        self.assertEqual(row["cache_snapshots"]["128"]["candidate"]["sequence_length"], 128)
        self.assertGreater(row["cache_snapshots"]["128"]["candidate"]["persistent_bytes"], 0)
        self.assertLess(row["cache_snapshots"]["128"]["candidate"]["persistent_bytes"],
                        row["cache_snapshots"]["128"]["teacher"]["persistent_bytes"])
        self.assertGreater(row["cache_snapshots"]["64"]["candidate"]["quantized_layers"], 0)

    @unittest.skipUnless(importlib.util.find_spec("hqq"), "optional HQQ backend is not installed")
    def test_batched_rows_match_independent_runs_and_keep_total_cache_bytes(self):
        torch.set_num_threads(2)
        torch.manual_seed(37)
        config = Qwen3Config(vocab_size=32, hidden_size=128, intermediate_size=160,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=64)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        tokens = torch.randint(0, 32, (2, 128))
        fields = ("teacher_nll", "candidate_nll", "whole_nll_increase", "teacher_kl",
                  "decode_teacher_nll", "decode_candidate_nll", "decode_nll_increase",
                  "decode_teacher_kl", "seam_target_nll_increase")
        for policy in ("identity", "hqq8", "hqq4"):
            batched = metrics.evaluate_batch(model, tokens, policy)
            self.assertEqual(len(batched), 2)
            for index, row in enumerate(batched):
                single = metrics.evaluate_block(model, tokens[index:index + 1], policy)
                self.assertEqual(row["batch_size"], 2)
                self.assertEqual(row["whole_nll_targets"], 127)
                self.assertEqual(row["decode_nll_targets"], 63)
                for field in fields:
                    self.assertAlmostEqual(row[field], single[field], delta=2e-4, msg=f"{policy} {field}")
                for length in metrics.SNAPSHOT_LENGTHS:
                    state = row["cache_snapshots"][str(length)]["candidate"]
                    solo = single["cache_snapshots"][str(length)]["candidate"]
                    self.assertEqual(state["batch_size"], 2)
                    self.assertEqual(state["persistent_bytes_per_sequence_derived"], state["persistent_bytes"] / 2)
                    self.assertAlmostEqual(state["persistent_bytes_per_sequence_derived"], solo["persistent_bytes"],
                                           delta=64)

    def test_gate_requires_whole_and_decode_quality(self):
        accepted = {"whole_nll_increase": .02, "whole_teacher_kl": .02,
                    "decode_nll_increase": .02, "decode_teacher_kl": .02}
        self.assertTrue(metrics.passes_gate(accepted))
        rejected = dict(accepted, decode_teacher_kl=.02001)
        self.assertFalse(metrics.passes_gate(rejected))
        self.assertFalse(metrics.passes_gate(dict(accepted, whole_nll_increase=float("nan"))))

    def test_selection_requires_quality_and_real_bytes_saved(self):
        base = {"passed": True, "whole_nll_increase": .01, "persistent_bytes": {"128": 80}}
        candidates = {"identity": {"teacher_persistent_bytes": {"128": 100}},
                      "hqq8": dict(base, persistent_bytes={"128": 90}),
                      "hqq4": dict(base, persistent_bytes={"128": 70})}
        self.assertEqual(metrics.choose_policy(candidates), "hqq4")
        candidates["hqq4"]["passed"] = False
        self.assertEqual(metrics.choose_policy(candidates), "hqq8")
        candidates["hqq8"]["persistent_bytes"] = {"128": 100}
        self.assertEqual(metrics.choose_policy(candidates), "identity")

    def test_confirmatory_freshness_audit_rejects_previously_opened_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence" / "quantized-validation-old"
            evidence.mkdir(parents=True)
            receipt = evidence / "hqq8-blocks.jsonl"
            receipt.write_text(json.dumps({"block_id": 127}) + "\n", encoding="utf-8")
            self.assertEqual(len(metrics._audit_unopened_tail(root, root / "selection.json")), 1)
            receipt.write_text(json.dumps({"block_id": 128}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already appears"):
                metrics._audit_unopened_tail(root, root / "selection.json")

    def test_confirmatory_freshness_audit_rejects_started_attempt_without_completed_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence" / "quantized-confirmatory-aborted"
            evidence.mkdir(parents=True)
            (evidence / "confirmatory-blocks.jsonl").write_text("", encoding="utf-8")
            metrics.record_batch_attempt(evidence / "confirmatory-attempts.jsonl", 0, range(128, 132))
            with self.assertRaisesRegex(ValueError, "tail was already opened"):
                metrics._audit_unopened_tail(root, root / "selection.json")

    def test_frozen_selection_digest_and_recomputed_choice_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation"
            validation.mkdir()
            candidates = {}
            for policy, passed, cache_bytes in (("identity", True, 100), ("hqq8", True, 80),
                                                ("hqq4", False, 60)):
                metrics_path = validation / f"{policy}-metrics.json"
                blocks_path = validation / f"{policy}-blocks.jsonl"
                delta = 0.0 if policy == "identity" else (0.01 if passed else 0.03)
                kl = 0.0 if policy == "identity" else (0.01 if passed else 0.03)
                rows = []
                lengths = ("64", "95", "96", "97", "127", "128")
                with blocks_path.open("w", encoding="utf-8") as stream:
                    for block_id in range(128):
                        batch_index = block_id // metrics.BATCH_SIZE
                        snapshots = {}
                        for length in lengths:
                            candidate_bytes = cache_bytes if length == "128" else cache_bytes + int(length)
                            teacher_bytes = 100 if length == "128" else 100 + int(length)
                            snapshots[length] = {}
                            for side, size in (("teacher", teacher_bytes), ("candidate", candidate_bytes)):
                                snapshots[length][side] = {
                                    "sequence_length": int(length), "batch_size": metrics.BATCH_SIZE,
                                    "persistent_bytes": size,
                                    "persistent_bytes_per_sequence_derived": size / metrics.BATCH_SIZE}
                        row = {"block_id": block_id, "batch_index": batch_index,
                               "batch_size": metrics.BATCH_SIZE,
                               "teacher_nll": 2.0, "candidate_nll": 2.0 + delta,
                               "teacher_kl": kl, "token_agreement": 1.0,
                               "decode_teacher_nll": 2.0, "decode_candidate_nll": 2.0 + delta,
                               "decode_teacher_kl": kl, "decode_token_agreement": 1.0,
                               "cache_snapshots": snapshots}
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                values = metrics.summarize(rows)
                metrics.write_json(metrics_path, values)
                candidates[policy] = {"blocks": 128, "passed": passed,
                                      "metrics_file": metrics_path.name,
                                      "metrics_sha256": metrics.file_hash(metrics_path),
                                      "blocks_file": blocks_path.name,
                                      "blocks_sha256": metrics.file_hash(blocks_path), "metrics": values}
            result = {"status": "complete", "action": "validate", "source_sha256": metrics.source_hashes(),
                      "policy_config": metrics.POLICY_CONFIG, "gates": metrics.GATES,
                      "schedule": metrics.SCHEDULE, "batch_size": metrics.BATCH_SIZE,
                      "coverage_blocks": {policy: 128 for policy in metrics.POLICIES},
                      "coverage_batches": {policy: 32 for policy in metrics.POLICIES},
                      "block_range": list(metrics.VALIDATION_RANGE), "candidates": candidates,
                      "runtime": {"python": "test", "torch": "test", "transformers": "test",
                                  "hqq": "test", "device": "cpu", "dtype": "torch.float32",
                                  "cuda_build": None, "device_name": None,
                                  "device_capability": None, "total_device_bytes": None},
                      "model_identity": {"revision": "pinned"}, "data_provenance_sha256": "data-hash",
                      "prepared_validation_sha256": {"validation/data.arrow": "prepared-hash"}}
            metrics.write_json(validation / "result.json", result)
            recomputed, receipts, selected = metrics.recompute_validation_choice(validation)
            self.assertEqual(selected, "hqq8")
            selection = {"status": "frozen", "confirmatory_evaluated": False, "selected": selected,
                         "selected_nbits": 8, "source_sha256": metrics.source_hashes(),
                         "policy_config": metrics.POLICY_CONFIG, "gates": metrics.GATES,
                         "schedule": metrics.SCHEDULE,
                         "validation_block_range": list(metrics.VALIDATION_RANGE),
                         "confirmatory_block_range": list(metrics.CONFIRMATORY_RANGE),
                         "validation_path": str(validation.resolve()), "validation_receipts": receipts,
                         "model_identity": recomputed["model_identity"],
                         "runtime": recomputed["runtime"],
                         "data_provenance_sha256": recomputed["data_provenance_sha256"],
                         "prepared_validation_sha256": recomputed["prepared_validation_sha256"]}
            selection_path = root / "selection.json"
            metrics.write_json(selection_path, selection)
            digest = metrics.file_hash(selection_path)
            metrics.verify_frozen_selection(selection_path, digest)
            selection["runtime"] = dict(selection["runtime"], torch="different-build")
            metrics.write_json(selection_path, selection)
            with self.assertRaisesRegex(ValueError, "recomputed validation decision"):
                metrics.verify_frozen_selection(selection_path, metrics.file_hash(selection_path))
            selection["runtime"] = recomputed["runtime"]
            metrics.write_json(selection_path, selection)
            digest = metrics.file_hash(selection_path)
            metrics.verify_frozen_selection(selection_path, digest)

            # Updating every receipt hash must not make fabricated aggregates valid.
            candidate = result["candidates"]["hqq8"]
            metrics_path = validation / candidate["metrics_file"]
            altered = json.loads(metrics_path.read_text(encoding="utf-8"))
            altered["persistent_bytes"]["128"] += 1
            metrics.write_json(metrics_path, altered)
            candidate["metrics"] = altered
            candidate["metrics_sha256"] = metrics.file_hash(metrics_path)
            metrics.write_json(validation / "result.json", result)
            with self.assertRaisesRegex(ValueError, "do not recompute"):
                metrics.read_validation_receipts(validation)
            # Restore the canonical metrics before the existing frozen-selection tamper checks.
            original_rows = [json.loads(line) for line in
                             (validation / candidate["blocks_file"]).read_text(encoding="utf-8").splitlines()]
            candidate["metrics"] = metrics.summarize(original_rows)
            metrics.write_json(metrics_path, candidate["metrics"])
            candidate["metrics_sha256"] = metrics.file_hash(metrics_path)
            metrics.write_json(validation / "result.json", result)
            recomputed, receipts, selected = metrics.recompute_validation_choice(validation)
            selection["validation_receipts"] = receipts
            selection["model_identity"] = recomputed["model_identity"]
            selection["runtime"] = recomputed["runtime"]
            metrics.write_json(selection_path, selection)
            digest = metrics.file_hash(selection_path)
            metrics.verify_frozen_selection(selection_path, digest)
            selection["selected"] = "hqq4"
            metrics.write_json(selection_path, selection)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                metrics.verify_frozen_selection(selection_path, digest)
            with self.assertRaisesRegex(ValueError, "recomputed validation decision"):
                metrics.verify_frozen_selection(selection_path, metrics.file_hash(selection_path))
            selection["selected"] = "hqq8"
            selection["selected_nbits"] = 4
            metrics.write_json(selection_path, selection)
            with self.assertRaisesRegex(ValueError, "recomputed validation decision"):
                metrics.verify_frozen_selection(selection_path, metrics.file_hash(selection_path))
            selection["selected_nbits"] = 8
            selection["confirmatory_block_range"] = [128, 385]
            metrics.write_json(selection_path, selection)
            with self.assertRaisesRegex(ValueError, "current frozen receipt"):
                metrics.verify_frozen_selection(selection_path, metrics.file_hash(selection_path))

    def test_resource_guard_has_fixed_timeout_and_reports_unmeasured_rss(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TimeoutError, "600 second budget"):
                metrics.resource_guard(time.monotonic() - 601, Path(temporary), "cpu", "validate")
        receipt = metrics.resource_receipt("cpu")
        self.assertIsNone(receipt["process_peak_rss_bytes"])
        self.assertEqual(receipt["process_peak_rss_status"], "not measured")


if __name__ == "__main__":
    unittest.main()
