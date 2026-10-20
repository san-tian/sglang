#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 先验证L=4096。下一档先重启D/M并将两端--context-length都改为新的L，
# 其余对照配置保持一致；新模型的档位使用其官方上限，第1章数字只对应当前Qwen模型。
export L="${L:-4096}"
python3 - <<'PYTHON'
import json, os
from pathlib import Path
from urllib.request import urlopen
from transformers import AutoTokenizer
L = int(os.environ['L'])
assert L > 256
root = Path('/work/mol/evidence')
dest = root/f'long-{L}'
assert not dest.exists(), '禁止覆盖已有长测'
dest.mkdir()
for side, port in [('D', 30000), ('M', 30002)]:
    with urlopen(f'http://127.0.0.1:{port}/get_server_info', timeout=30) as f:
        info = json.load(f)
    (dest/f'{side}-server-info.json').write_text(json.dumps(info, indent=2))
    args = info.get('server_args', info)
    assert args['context_length'] == L, f'{side}尚未按当前L重启'
    assert not args.get('allow_auto_truncate', False), '必须关闭自动截断'
corpus = json.loads((root/'corpus.json').read_text())
assert len(corpus) >= 10
tok = AutoTokenizer.from_pretrained('/work/mol/base')
rows = []
for i, case in enumerate(corpus[:10]):
    prefix = tok.encode(f'Long-context case {i}: ', add_special_tokens=False)
    seed = case['input_ids']
    assert seed and len(prefix) < L-256
    n = L-256-len(prefix)
    ids = prefix + (seed*((n+len(seed)-1)//len(seed)))[:n]
    assert len(ids) == L-256
    rows.append({'id': f'L{L}-case{i}', 'input_ids': ids})
assert len({tuple(r['input_ids']) for r in rows}) == 10
(dest/'corpus.json').write_text(json.dumps(rows))
print('cases=10', 'input_tokens=',L-256,'output_tokens=256','total_tokens=',L)
PYTHON
set -e
python3 benchmark/lora/compare_merged.py   --dynamic-url http://127.0.0.1:30000 --merged-url http://127.0.0.1:30002   --adapter-name A --adapter-path /work/mol/adapter-A   --corpus /work/mol/evidence/long-${L}/corpus.json   --capture-dir /work/mol/evidence/long-${L}/captures   --output /work/mol/evidence/long-${L}/comparison   --limit 10 --decode-tokens 256 --rounds 3
python3 - <<'PYTHON'
import json, math, os
from pathlib import Path
L = int(os.environ['L'])
root = Path(f'/work/mol/evidence/long-{L}')
s = json.loads((root/'comparison/summary.json').read_text())
cases = json.loads((root/'corpus.json').read_text())
rows = [json.loads(x) for x in (root/'comparison/cases.jsonl').read_text().splitlines()]
expected = {(c['id'], r, mode) for c in cases for r in range(3) for mode in ('cold','hot')}
assert len(rows) == len(expected) == 60
assert {(r['case'],r['round'],r['cache']) for r in rows} == expected
assert (s['cases'],s['rounds'],s['decode_tokens']) == (10,3,256)
assert s['prefill_logprobs_measured']
def gate(row):
    assert row['count'] > 0 and row['passes_logprob_gate'] is True
    assert all(math.isfinite(row[k]) for k in ('mean_abs_nat','p99_abs_nat','max_abs_nat'))
    assert row['mean_abs_nat'] <= 0.01 and row['p99_abs_nat'] <= 0.1
for phase, count in {'prefill':60*(L-257),'prefill_last':60,'decode':15300}.items():
    assert s['scores'][phase]['count'] == count
    gate(s['scores'][phase])
    for row in rows:
        assert row['prompt_tokens'] == L-256 and row['continuation_tokens'] == 256
        gate(row[phase])
    for side in ('D','M'):
        gate(s['repeat_noise'][side][phase])
    print(phase, s['scores'][phase])
print('当前L的单adapter/串行/全prefill对拍通过；混批、真实缓存恢复和任务质量另验')
PYTHON
python3 - <<'PYTHON'
import json, os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
L = int(os.environ['L'])
root = Path(f'/work/mol/evidence/long-{L}')
ids = json.loads((root/'corpus.json').read_text())[0]['input_ids']
def post(port, body):
    req = Request(f'http://127.0.0.1:{port}/generate',
                  data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    try:
        with urlopen(req, timeout=7200) as f:
            return f.status, f.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()
for side, port in [('D',30000),('M',30002)]:
    for label, tokens, out in [('total-L-plus-1',ids,257),
                               ('input-L-plus-1',(ids*2)[:L+1],1),
                               ('short-after-errors',ids[:16],16)]:
        req = {'input_ids':tokens, 'sampling_params':{'temperature':0,'max_new_tokens':out,'ignore_eos':True}}
        if side == 'D':
            req['lora_path'] = 'A'
        status, response = post(port, req)
        (root/f'{side}-{label}.json').write_text(json.dumps(
            {'status':status,'input_tokens':len(tokens),'max_new_tokens':out,'response':response},indent=2))
        if label == 'short-after-errors':
            assert status == 200 and len(json.loads(response)['output_ids']) == 16
        else:
            assert status == 400, (side,label,status,response)
print('边界通过：D/M共4次超限明确400，随后2次短请求各完整生成16 tokens')
PYTHON
curl --fail --silent --show-error http://127.0.0.1:30000/unload_lora_adapter   -H 'Content-Type: application/json' -d '{"lora_name":"A"}'
