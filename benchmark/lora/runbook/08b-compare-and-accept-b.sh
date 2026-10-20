#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3；确认终端2已显示M-B启动成功。
set -e
curl --fail --silent --show-error --retry 120 --retry-connrefused --retry-delay 2 \
  http://127.0.0.1:30002/health -o /dev/null
curl --fail --silent --show-error http://127.0.0.1:30002/get_server_info > /work/mol/evidence/M-B-server-info.json
curl --fail --silent --show-error http://127.0.0.1:30002/model_info > /work/mol/evidence/M-B-model-info.json
python3 /work/mol/evidence/check-merge.py /work/mol/merged-B /work/mol/adapter-B
python3 - <<'PYTHON'
import json
from pathlib import Path
root=Path('/work/mol/evidence')
model=json.loads((root/'M-B-model-info.json').read_text())
assert Path(model['model_path']).resolve()==Path('/work/mol/merged-B').resolve(), '终端2尚未切到M-B'
info=json.loads((root/'M-B-server-info.json').read_text())
args=info.get('server_args',info)
assert 'enable_lora' in args and (args['enable_lora'] is None or args['enable_lora'] is False), 'M参考必须关闭LoRA'
print('B参考的输入/分片哈希、模型路径和禁用LoRA检查通过')
PYTHON
python3 benchmark/lora/compare_merged.py \
  --dynamic-url http://127.0.0.1:30000 --merged-url http://127.0.0.1:30002 \
  --adapter-name B --adapter-path /work/mol/adapter-B \
  --corpus /work/mol/evidence/corpus.json --capture-dir /work/mol/evidence/B-captures \
  --output /work/mol/evidence/main-B --limit 30 --decode-tokens 256 --rounds 4
python3 - <<'PYTHON'
import json, math
from pathlib import Path
p=Path('/work/mol/evidence')
s=json.loads((p/'main-B/summary.json').read_text())
corpus=json.loads((p/'corpus.json').read_text())
rows=[json.loads(x) for x in (p/'main-B/cases.jsonl').read_text().splitlines()]
assert (s['cases'],s['rounds'],s['decode_tokens'])==(30,4,256)
assert s['prefill_logprobs_measured'] and s['adapter']=='B'
expected={(c['id'],r,m) for c in corpus for r in range(4) for m in ('cold','hot')}
assert len(rows)==len(expected)==240
assert {(r['case'],r['round'],r['cache']) for r in rows}==expected
counts={'prefill':sum(len(c['input_ids'])-1 for c in corpus)*8,'prefill_last':240,'decode':61200}
for phase,count in counts.items():
    assert s['scores'][phase]['count']==count
    groups=[s['scores'][phase]]+[r[phase] for r in rows]+[s['repeat_noise'][side][phase] for side in ('D','M')]
    for g in groups:
        assert g['count']>0 and g['passes_logprob_gate'] is True
        assert all(math.isfinite(g[k]) for k in ('mean_abs_nat','p99_abs_nat','max_abs_nat'))
        assert g['mean_abs_nat']<=0.01 and g['p99_abs_nat']<=0.1
    print(phase,s['scores'][phase])
print('B独立merge参考：240对逐例及重复噪声通过')
PYTHON
curl --fail --silent --show-error http://127.0.0.1:30000/unload_lora_adapter \
  -H 'Content-Type: application/json' -d '{"lora_name":"B"}'
