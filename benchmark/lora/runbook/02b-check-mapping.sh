#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3；第2步两份CSV人工核对并填写后执行。
python3 - <<'PYTHON'
import csv,json
from pathlib import Path
r=Path('/work/mol/evidence')
for label in ('A','B'):
    with (r/f'target-map-{label}.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert rows, label
    keys=[row['checkpoint_key'] for row in rows]
    assert len(keys)==len(set(keys)), f'{label}存在重复checkpoint_key'
    for row in rows:
        for field in ('runtime_layer','packed_slice','tp_shard'):
            value=row[field].strip()
            assert value and value.lower() not in ('todo','待填写','待补'), (label,row['checkpoint_key'],field)
        a,b=json.loads(row['A_shape']),json.loads(row['B_shape'])
        rank=int(row['rank'])
        assert len(a)==len(b)==2 and a[0]==b[1]==rank
    print(label,'映射表填写检查通过；投影数=',len(rows))
print('这里只检查表格完整性；实际前向、融合切片和TP分片仍按第2步核对。')
PYTHON
