#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
cat > /work/mol/evidence/adapters.json <<'JSON'
[{"name":"A","path":"/work/mol/adapter-A"},{"name":"B","path":"/work/mol/adapter-B"},{"name":"zero","path":"/work/mol/adapter-zero"}]
JSON
python3 benchmark/lora/check_lifecycle.py \
  --url http://127.0.0.1:30000 \
  --adapters /work/mol/evidence/adapters.json \
  --corpus /work/mol/evidence/corpus.json \
  --output /work/mol/evidence/lifecycle \
  --cycles 100 --bad-loads 100 --aliases 12 --lru-rounds 9
python3 - <<'PYTHON'
import json
from pathlib import Path
s = json.loads(Path('/work/mol/evidence/lifecycle/summary.json').read_text())
assert s['passed'] and s['serial_cycles'] == s['invalid_loads'] == 100
assert s['lru_requests'] == 108 and s['inflight_output_tokens'] == 256
assert s['new_requests_rejected_while_draining'] and s['max_repeat_logprob_abs_nat'] == 0
print('生命周期检查通过：100换权重 + 100错误加载 + 108 LRU + 256-token在途卸载')
PYTHON
