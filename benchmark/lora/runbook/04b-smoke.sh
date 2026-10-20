#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3；终端1中的D持续运行。每个HTTP响应和状态都保存到smoke-api。
set -e
curl --fail --silent --show-error --retry 120 --retry-connrefused --retry-delay 2 \
  http://127.0.0.1:30000/health -o /dev/null
curl --fail --silent --show-error http://127.0.0.1:30000/get_server_info > /work/mol/evidence/D-server-info.json
curl --fail --silent --show-error http://127.0.0.1:30000/model_info > /work/mol/evidence/D-model-info.json
python3 - <<'PYTHON'
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
out=Path('/work/mol/evidence/smoke-api')
out.mkdir(exist_ok=False)
def call(label, route, body):
    req=Request('http://127.0.0.1:30000'+route, data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json'})
    try:
        with urlopen(req, timeout=300) as f: status, raw=f.status, f.read().decode()
    except HTTPError as e:
        status, raw=e.code, e.read().decode()
    (out/(label+'.json')).write_text(json.dumps({'status':status,'response':raw},indent=2))
    assert status==200, (label,status,raw)
    return json.loads(raw)
def generate(name):
    body={'text':'Hello','sampling_params':{'temperature':0,'max_new_tokens':16,'ignore_eos':True},
          'return_logprob':True}
    if name!='base': body['lora_path']=name
    result=call(name+'-generate','/generate',body)
    assert len(result['output_ids'])==16
    assert len(result['meta_info']['output_token_logprobs'])==16
    return result
base=generate('base')
for name, folder in [('A','adapter-A'),('B','adapter-B'),('zero','adapter-zero')]:
    call(name+'-load','/load_lora_adapter',{'lora_name':name,'lora_path':'/work/mol/'+folder})
    try:
        result=generate(name)
        if name=='zero':
            assert result['output_ids']==base['output_ids']
            assert result['meta_info']['output_token_logprobs']==base['meta_info']['output_token_logprobs']
    finally:
        call(name+'-unload','/unload_lora_adapter',{'lora_name':name})
    print(name, 'load/generate/unload passed; output_tokens=16')
print('base/A/B/zero均完成；zero与base的自然输出及16个logprob一致')
PYTHON
