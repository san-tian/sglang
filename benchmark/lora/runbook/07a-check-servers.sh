#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
set -e
curl --fail --silent --show-error --retry 120 --retry-connrefused --retry-delay 2 \
  http://127.0.0.1:30002/health -o /dev/null
curl --fail --silent --show-error http://127.0.0.1:30002/get_server_info > /work/mol/evidence/M-server-info.json
curl --fail --silent --show-error http://127.0.0.1:30002/model_info > /work/mol/evidence/M-model-info.json
python3 - <<'PYTHON'
import json
from pathlib import Path
from urllib.request import urlopen
root = Path('/work/mol/evidence')
configs = {}
for side, port, expected in [('D',30000,'/work/mol/base'),('M',30002,'/work/mol/merged-A')]:
    with urlopen(f'http://127.0.0.1:{port}/model_info',timeout=30) as f:
        model = json.load(f)
    assert model['model_path'] == expected, (side,model)
    with urlopen(f'http://127.0.0.1:{port}/get_server_info',timeout=30) as f:
        info = json.load(f)
    (root/f'{side}-before-comparison-server-info.json').write_text(json.dumps(info,indent=2))
    configs[side] = info.get('server_args',info)
keys = ('dtype','tp_size','context_length','rope_scaling','attention_backend','linear_attn_prefill_backend',
        'bf16_gemm_backend','disable_cuda_graph','enable_deterministic_inference')
for key in keys:
    assert key in configs['D'] and key in configs['M'], f'响应缺少{key}；先核对当前版本字段'
    assert configs['D'][key] == configs['M'][key], (key,configs['D'][key],configs['M'][key])
assert configs['D']['tp_size'] == 1 and configs['D']['context_length'] == 4096
assert configs['D']['enable_lora'] and not configs['M']['enable_lora']
print('D=base+动态adapter；M=merged-A；两端关键配置一致，可以开始短测')
PYTHON
