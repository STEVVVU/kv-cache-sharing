"""Dependency-free checks for validation gates and the final-test receipt boundary."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "trainer_metrics.py"
spec = importlib.util.spec_from_file_location("trainer_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def grid():
    return [{"strategy": strategy, "pairs": pairs, "seed": 11, "study": copy.deepcopy(metrics.STUDY),
             "pilot_only": False, "test_evaluated": False, "passed": True, "global_step": 256,
             "exact_checkpoint_reload": True, "source_sha256": {"driver": "same"},
             "model_identity": {"revision": "same"}, "data_provenance_sha256": "same",
             "validation": {"blocks": 128, "mean_paired_nll_increase": 0.01, "teacher_kl": 0.01,
                            "unique_cache_bytes": 2800 - 100 * pairs, "baseline_cache_bytes": 2800,
                            "next_token_nll": 3.01, "baseline_nll": 3.0}}
            for strategy in metrics.STUDY["strategies"] for pairs in metrics.STUDY["pairs"]]


class MetricsContractTests(unittest.TestCase):
    def test_pairing_and_spread(self):
        rows = [{"block_id": i, "next_token_nll": 3 + delta, "baseline_nll": 3,
                 "teacher_kl": 0.01, "token_agreement": 0.9, "unique_cache_bytes": 2600,
                 "baseline_cache_bytes": 2800} for i, delta in enumerate((-0.1, 0.1))]
        result = metrics.summarize(rows)
        self.assertAlmostEqual(result["mean_paired_nll_increase"], 0)
        self.assertGreater(result["paired_nll_delta_stdev"], 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            metrics.summarize([rows[0], rows[0]])

    def test_gate_rejects_quality_failure_and_nonfinite(self):
        self.assertTrue(metrics.passes_gate({"mean_paired_nll_increase": 0.02, "teacher_kl": 0.02}))
        for nll, kl in ((0.02001, 0.01), (0.01, 0.02001), (float("nan"), 0.0)):
            self.assertFalse(metrics.passes_gate({"mean_paired_nll_increase": nll, "teacher_kl": kl}))

    def test_selection_prefers_storage_only_after_gate(self):
        rows = grid()
        for row in rows:
            if row["pairs"] == 4:
                row["validation"]["mean_paired_nll_increase"] = 1
        selected = metrics.choose_candidate(rows)
        self.assertEqual((selected["strategy"], selected["pairs"]), ("cka", 2))
        for row in rows:
            row["validation"]["teacher_kl"] = 1
        self.assertIsNone(metrics.choose_candidate(rows))

    def test_incomplete_pilot_or_test_contamination_cannot_select(self):
        with self.assertRaisesRegex(ValueError, "six distinct"):
            metrics.choose_candidate(grid()[:-1])
        for key in ("pilot_only", "test_evaluated"):
            rows = grid()
            rows[0][key] = True
            with self.assertRaisesRegex(ValueError, "incomplete"):
                metrics.choose_candidate(rows)
        rows = grid()
        rows[0]["validation"]["baseline_nll"] += 0.01
        with self.assertRaisesRegex(ValueError, "unmatched independent"):
            metrics.choose_candidate(rows)

    def test_final_receipt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = []
            rows = []
            for seed in metrics.STUDY["seeds"]:
                for row in grid():
                    row["seed"] = seed
                    rows.append(row)
            for i, row in enumerate(rows):
                row["validation"]["teacher_kl"] = 1  # Freeze an identity fallback.
                path = root / f"candidate-{i}.json"
                metrics.write_json(path, row)
                receipts.append({"path": str(path), "sha256": metrics.file_hash(path)})
            selection = root / "selection.json"
            metrics.write_json(selection, {"selected": {str(seed): None for seed in metrics.STUDY["seeds"]},
                                          "candidate_receipts": receipts, "study": metrics.STUDY,
                                          "test_evaluated": False})
            self.assertIsNone(metrics.load_selection(selection)[0])
            Path(receipts[0]["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed after selection"):
                metrics.load_selection(selection)

    def test_one_seed_cannot_open_final_test(self):
        with self.assertRaisesRegex(ValueError, "all five seeds"):
            metrics.choose_study(grid())


if __name__ == "__main__":
    unittest.main()
