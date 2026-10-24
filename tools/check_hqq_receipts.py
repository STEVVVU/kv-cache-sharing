"""Verify LF publication copies against original frozen HQQ receipts, without a GPU."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import quantized_metrics as metrics

SELECTION_SHA256 = "600632d7e8603596e8dd01629fd8f6db0c8a7a636ca7ef923661cccdb7d0b7fb"
RAW_FILES_SHA256 = "d10d222c929590125e9810e72be8c2146257fd7f8b21d79950e7bed935b3c167"
SOURCE_SHA256 = {
    "kvshare.py": "7d9c6138e40aff4c7ab2c8bdca87ba1cfafad48b68c1d6e08a4f44c1601029ba",
    "tools\\quantized_metrics.py": "d1de0653556906827fc5753bbc0f4a792ca29e3680eab81d3251ada7b8c4ec17",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source_binding(source_root):
    for name, expected in SOURCE_SHA256.items():
        data = (Path(source_root) / name.replace("\\", "/")).read_bytes()
        # Source identity allows only Git's LF/CRLF checkout conversion.
        canonical = data.replace(b"\r\n", b"\n")
        if expected not in {hashlib.sha256(value).hexdigest()
                            for value in (canonical, canonical.replace(b"\n", b"\r\n"))}:
            raise ValueError(f"current source binding mismatch: {name}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_raw_receipts(root):
    validation = root / "validation"
    result = read_json(validation / "result.json")
    selection_path = root / "selection" / "selection.json"
    selection = read_json(selection_path)
    confirmation = root / "confirmatory"
    confirmed = read_json(confirmation / "result.json")
    for receipt in (result, selection, confirmed):
        if (receipt["source_sha256"] != SOURCE_SHA256
                or receipt["policy_config"] != metrics.POLICY_CONFIG
                or receipt["schedule"] != metrics.SCHEDULE or receipt["gates"] != metrics.GATES):
            raise ValueError("frozen source, policy, schedule or gates changed")
    for key in ("runtime", "model_identity", "data_provenance_sha256", "prepared_validation_sha256"):
        if not result[key] == selection[key] == confirmed[key]:
            raise ValueError(f"frozen study identity changed: {key}")
    if (result["status"] != "complete" or result["action"] != "validate"
            or result["block_range"] != [0, 128] or result["batch_size"] != 4
            or result["coverage_blocks"] != dict.fromkeys(metrics.POLICIES, 128)
            or result["coverage_batches"] != dict.fromkeys(metrics.POLICIES, 32)):
        raise ValueError("validation action or coverage changed")
    receipts = {"result.json": digest(validation / "result.json")}
    summaries = {}
    baseline = None
    for policy in metrics.POLICIES:
        candidate = result["candidates"][policy]
        for field in ("blocks", "metrics"):
            path = validation / candidate[f"{field}_file"]
            if digest(path) != candidate[f"{field}_sha256"]:
                raise ValueError("validation receipt hash mismatch")
            receipts[path.name] = digest(path)
        rows = [json.loads(line) for line in
                (validation / candidate["blocks_file"]).read_text(encoding="utf-8").splitlines()]
        if [row["block_id"] for row in rows] != list(range(128)) or candidate["blocks"] != 128:
            raise ValueError("validation block coverage changed")
        summaries[policy] = metrics.summarize(rows)
        if (summaries[policy] != candidate["metrics"]
                or summaries[policy] != read_json(validation / candidate["metrics_file"])
                or summaries[policy]["passed"] != candidate["passed"]):
            raise ValueError("validation aggregate does not recompute from rows")
        teachers = [(row["teacher_nll"], row["cache_snapshots"]["128"]["teacher"]) for row in rows]
        if baseline is not None and teachers != baseline:
            raise ValueError("validation teacher baseline changed")
        baseline = teachers
    selected = metrics.choose_policy(summaries)
    if (digest(selection_path) != SELECTION_SHA256 or selection["status"] != "frozen"
            or selection["action"] != "select" or selection["confirmatory_evaluated"]
            or selection["selected"] != selected or selected != "hqq8"
            or selection["selected_nbits"] != metrics.POLICIES[selected]
            or selection["validation_receipts"] != receipts
            or selection["validation_block_range"] != [0, 128]
            or selection["confirmatory_block_range"] != [128, 384]):
        raise ValueError("frozen selection does not match original validation decision")
    if (confirmed["action"] != "confirmatory" or confirmed["status"] != "complete"
            or confirmed["selection"] != selection or confirmed["selection_sha256"] != SELECTION_SHA256
            or confirmed["selected"] != selected or confirmed["confirmatory_block_range"] != [128, 384]
            or confirmed["coverage_blocks"] != 256 or confirmed["coverage_batches"] != 64):
        raise ValueError("confirmatory frozen selection or coverage changed")
    for field in ("blocks", "metrics"):
        if digest(confirmation / confirmed[f"{field}_file"]) != confirmed[f"{field}_sha256"]:
            raise ValueError("confirmatory receipt hash mismatch")
    rows = [json.loads(line) for line in
            (confirmation / confirmed["blocks_file"]).read_text(encoding="utf-8").splitlines()]
    if [row["block_id"] for row in rows] != list(range(128, 384)):
        raise ValueError("confirmatory block coverage changed")
    if (metrics.summarize(rows) != confirmed["metrics"]
            or confirmed["metrics"] != read_json(confirmation / confirmed["metrics_file"])):
        raise ValueError("confirmatory aggregate does not recompute from rows")
    attempts = [json.loads(line) for line in
                (confirmation / "confirmatory-attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    time = lambda value: datetime.fromisoformat(value)
    if not (time(result["finished_at_utc"]) <= time(selection["started_at_utc"])
            <= time(selection["created_at_utc"]) <= time(confirmed["started_at_utc"])):
        raise ValueError("validation/selection/confirmation freeze ordering changed")
    previous = time(confirmed["started_at_utc"])
    if len(attempts) != 64:
        raise ValueError("confirmatory attempt coverage changed")
    for index, attempt in enumerate(attempts):
        started = time(attempt["started_at_utc"])
        if (attempt["batch_index"] != index
                or attempt["block_ids"] != list(range(128 + index * 4, 132 + index * 4))
                or not previous <= started <= time(confirmed["finished_at_utc"])):
            raise ValueError("confirmatory attempt coverage or ordering changed")
        previous = started
    return {"validation_blocks_per_candidate": 128, "selected": selected,
            "confirmatory_blocks": len(rows), "confirmatory_batches": len(attempts)}


def check_receipts(root, source_root=ROOT):
    root = Path(root).resolve()
    check_source_binding(source_root)
    manifest = read_json(root / "manifest.json")
    raw_hashes = {name: entry["raw_sha256"] for name, entry in manifest["files"].items()}
    if hashlib.sha256(json.dumps(raw_hashes, sort_keys=True).encode()).hexdigest() != RAW_FILES_SHA256:
        raise ValueError("original raw receipt hash manifest changed")
    expected = {f"validation/{policy}-{kind}.{extension}"
                for policy in metrics.POLICIES for kind, extension in (("blocks", "jsonl"), ("metrics", "json"))}
    expected |= {"validation/result.json", "selection/selection.json", "confirmatory/result.json",
                 "confirmatory/confirmatory-blocks.jsonl", "confirmatory/confirmatory-metrics.json",
                 "confirmatory/confirmatory-attempts.jsonl"}
    if set(manifest["files"]) != expected:
        raise ValueError("publication receipt set changed")
    with tempfile.TemporaryDirectory(prefix="hqq-publication-") as temporary:
        workspace = Path(temporary)
        for relative, entry in manifest["files"].items():
            data = (root / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["canonical_sha256"] or b"\r" in data:
                raise ValueError(f"canonical receipt hash or LF mismatch: {relative}")
            if entry["source_line_endings"] != "CRLF":
                raise ValueError(f"unsupported raw line endings: {relative}")
            raw = data.replace(b"\n", b"\r\n")
            if hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
                raise ValueError(f"raw receipt identity mismatch: {relative}")
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return check_raw_receipts(workspace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs" / "evidence" / "hqq-cache-20261024")
    args = parser.parse_args()
    print(json.dumps(check_receipts(args.evidence), sort_keys=True))


if __name__ == "__main__":
    main()
