#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
set -e
cat > /work/mol/evidence/adapters.json <<'JSON'
[{"name":"A","path":"/work/mol/adapter-A"},{"name":"B","path":"/work/mol/adapter-B"},{"name":"zero","path":"/work/mol/adapter-zero"}]
JSON
python3 benchmark/lora/compare_mixed.py \
  --url http://127.0.0.1:30000 --adapters /work/mol/evidence/adapters.json \
  --corpus /work/mol/evidence/corpus.json --capture-dir /work/mol/evidence/mixed-captures \
  --output /work/mol/evidence/mixed --cases 4 --decode-tokens 16 --rounds 3
python3 - <<'PYTHON'
import json, math
from pathlib import Path
s = json.loads(Path('/work/mol/evidence/mixed/summary.json').read_text())
assert s['adapters_including_base'] == 4 and s['isolated_requests'] == 16
assert s['mixed_requests'] == len(s['cases']) == 48
expected = {(f'{name}-case-{i:02d}', r) for name in ('base','A','B','zero') for i in range(4) for r in range(3)}
assert {(row['id'],row['round']) for row in s['cases']} == expected
for phase, count in (('prefill_last',48),('decode',720)):
    aggregate = s['scores'][phase]
    assert aggregate['count'] == count
    for row in [aggregate]+[case[phase] for case in s['cases']]:
        assert row['count'] > 0 and row['passes_logprob_gate'] is True
        assert all(math.isfinite(row[k]) for k in ('mean_abs_nat','p99_abs_nat','max_abs_nat'))
        assert row['mean_abs_nat'] <= 0.01 and row['p99_abs_nat'] <= 0.1
    print(phase, aggregate)
print('混流48请求逐例通过：48个首token + 720个decode评分；D/M精度仍单独验收')
PYTHON
