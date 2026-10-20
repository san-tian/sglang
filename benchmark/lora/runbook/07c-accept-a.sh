#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3；A主测完成后立即验收。此时不要冻结目录。
python3 - <<'PYTHON'
import json
from pathlib import Path
r = Path('/work/mol/evidence')
s = json.loads((r/'main/summary.json').read_text())
corpus = json.loads((r/'corpus.json').read_text())
rows = [json.loads(line) for line in (r/'main/cases.jsonl').read_text().splitlines()]
assert (s['cases'], s['rounds'], s['decode_tokens']) == (30, 4, 256)
assert set(s['cache_modes']) == {'cold','hot'} and s['prefill_logprobs_measured']
expected = {(c['id'], n, mode) for c in corpus for n in range(4) for mode in ('cold','hot')}
actual = {(c['case'], c['round'], c['cache']) for c in rows}
assert len(rows) == len(actual) == len(expected) == 240 and actual == expected
counts = {'prefill': sum(len(c['input_ids'])-1 for c in corpus)*8,
          'prefill_last': 240, 'decode': 240*255}
def gate(row):
    assert row['count'] > 0 and row['passes_logprob_gate'] is True
    assert row['mean_abs_nat'] <= 0.01 and row['p99_abs_nat'] <= 0.1
for phase, expected_count in counts.items():
    aggregate = s['scores'][phase]
    assert aggregate['count'] == expected_count, (phase, aggregate)
    gate(aggregate)
    for row in rows:
        gate(row[phase])
    for side in ('D','M'):
        gate(s['repeat_noise'][side][phase])
    print(phase, 'count=', aggregate['count'], 'max=', aggregate['max_abs_nat'],
          'argmax=', aggregate['argmax_match_fraction'])
print('主矩阵240对逐例及重复噪声检查通过；生命周期、缓存、长上下文仍须分别验收')
PYTHON
