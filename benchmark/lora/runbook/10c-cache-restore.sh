#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
python3 benchmark/lora/check_cache_restore.py \
  --dynamic-url http://127.0.0.1:30000 \
  --merged-url http://127.0.0.1:30002 \
  --adapter-path /work/mol/adapter-A \
  --corpus /work/mol/evidence/corpus.json \
  --output /work/mol/evidence/cache-restore
python3 - <<'PYTHON'
import json
from pathlib import Path
s = json.loads(Path('/work/mol/evidence/cache-restore/summary.json').read_text())
assert s['passed'] and len(s['rows']) == 4
assert all(r['cached_tokens'] > 0 for r in s['rows'] if r['label'].endswith('-hot'))
print('D/M真实前缀命中检查通过；仍需另测跨adapter隔离与全prefill')
PYTHON
