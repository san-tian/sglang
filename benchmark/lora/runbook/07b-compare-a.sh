#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
set -e
python3 benchmark/lora/compare_merged.py --dynamic-url http://127.0.0.1:30000 --merged-url http://127.0.0.1:30002 --adapter-name A --adapter-path /work/mol/adapter-A --corpus /work/mol/evidence/corpus.json --capture-dir /work/mol/evidence/smoke-captures --output /work/mol/evidence/smoke --limit 4 --decode-tokens 16 --rounds 1
python3 - <<'PYTHON'
import json
from pathlib import Path
s = json.loads(Path('/work/mol/evidence/smoke/summary.json').read_text())
assert (s['cases'], s['rounds'], s['decode_tokens']) == (4, 1, 16)
for name in ('prefill', 'prefill_last', 'decode'):
    r = s['scores'][name]
    assert r['count'] > 0 and r['mean_abs_nat'] <= 0.01 and r['p99_abs_nat'] <= 0.1, (name, r)
print('smoke gates passed')
PYTHON
curl --fail --silent --show-error http://127.0.0.1:30000/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"A"}'
python3 benchmark/lora/compare_merged.py --dynamic-url http://127.0.0.1:30000 --merged-url http://127.0.0.1:30002 --adapter-name A --adapter-path /work/mol/adapter-A --corpus /work/mol/evidence/corpus.json --capture-dir /work/mol/evidence/main-captures --output /work/mol/evidence/main --limit 30 --decode-tokens 256 --rounds 4
curl --fail --silent --show-error http://127.0.0.1:30000/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"A"}'
