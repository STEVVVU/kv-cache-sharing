# KV cache sharing

Single-file Python library and CLI against Transformers 5.20.2 Qwen3 only.
Run `python tools/verify.py doctor` then `python tools/verify.py drive`.
Use `.agents/skills/verify-kv-cache-sharing/SKILL.md` for actual pretrained proof.
Do not download weights implicitly, enable remote code, or overlap GPU jobs.
Maintain fixtures.json train/calibration/heldout separation. The metrics study may explicitly prepare pinned public WikiText2 as documented in the metrics feature map; preserve its license, revision, hashes and official split boundaries. No implicit external dataset downloads.
Cache byte metrics count unique tensor storage, not whole-process GPU memory.
No benchmark-quality preservation claim follows from the synthetic smoke run.
