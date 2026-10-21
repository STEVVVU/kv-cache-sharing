"""Bounded real CLI driver; evidence survives subprocess cleanup."""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('action', choices=['doctor', 'drive', 'pretrained'])
p.add_argument('--timeout', type=int, default=600)
a = p.parse_args()
out = ROOT / 'evidence'
out.mkdir(exist_ok=True)
commands = {
    'doctor': [sys.executable, '-c', 'import torch, transformers; print(torch.__version__, transformers.__version__); print("cuda", torch.cuda.is_available())'],
    'drive': [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    'pretrained': [sys.executable, '-m', 'kvshare', 'experiment', '--device', 'cuda', '--output', 'evidence/pretrained.json'],
}
start = time.time()
try:
    run = subprocess.run(commands[a.action], cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
    result = dict(command=commands[a.action], exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
except subprocess.TimeoutExpired as e:
    result = dict(command=commands[a.action], exit_code=124, error=str(e))
result['elapsed_seconds'] = time.time() - start
(out / (a.action + '-driver.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
sys.exit(result['exit_code'])
