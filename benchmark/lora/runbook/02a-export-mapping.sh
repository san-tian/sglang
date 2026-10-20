#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
python3 - <<'PYTHON'
import csv, json
from pathlib import Path
from safetensors import safe_open
root = Path('/work/mol')
index = json.loads((root/'base/model.safetensors.index.json').read_text())['weight_map']
base_shapes = {}
for shard in sorted(set(index.values())):
    with safe_open(root/'base'/shard, framework='pt', device='cpu') as f:
        base_shapes.update({k: list(f.get_slice(k).get_shape()) for k in f.keys()})
(root/'evidence/base-shapes.json').write_text(json.dumps(base_shapes, indent=2))
for label in ('A', 'B'):
    cfg = json.loads((root/f'adapter-{label}/adapter_config.json').read_text())
    with safe_open(root/f'adapter-{label}/adapter_model.safetensors', framework='pt', device='cpu') as f:
        shapes = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
    used, rows = set(), []
    for a, a_shape in sorted(shapes.items()):
        if '.lora_A.' not in a:
            continue
        b = a.replace('.lora_A.', '.lora_B.')
        assert b in shapes, f'缺少配对B: {a}'
        b_shape = shapes[b]
        assert len(a_shape) == len(b_shape) == 2
        assert a_shape[0] == b_shape[1] == cfg['r'], (a, a_shape, b_shape)
        rows.append([a, '', json.dumps(a_shape), json.dumps(b_shape), cfg['r'], cfg['lora_alpha'], '', ''])
        used.update((a, b))
    assert rows and used == set(shapes), f'未消费的adapter键: {set(shapes)-used}'
    path = root/f'evidence/target-map-{label}.csv'
    assert not path.exists(), '保留已有映射；新实验使用新目录'
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint_key','runtime_layer','A_shape','B_shape','rank','alpha','packed_slice','tp_shard'])
        w.writerows(rows)
    print('adapter=',label,'adapter_projection_pairs=', len(rows), 'base_tensors=', len(base_shapes))
    print('下一步填写',path,'的runtime_layer/packed_slice/tp_shard，再进行代码适配')
PYTHON
# 在仓库根目录定位已有实现；重点看前向和切片，不只看类名。
rg -n 'supported_lora_modules|packed_modules_mapping|qkv_proj|gate_up_proj' python/sglang/srt/models
rg -n 'lora_A|lora_B|qkv_proj|gate_up_proj' python/sglang/srt/lora
