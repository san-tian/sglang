#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
cat > /work/mol/evidence/check-merge.py <<'PYTHON'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
expected_adapter=Path(sys.argv[2]).resolve()
assert (p/'MERGE_COMPLETE').is_file(), 'merge未完整结束'
s=json.loads((p/'merge-manifest.json').read_text())
assert Path(s['base']).resolve()==Path('/work/mol/base').resolve()
assert Path(s['adapter']).resolve()==expected_adapter, '参考权重对应的adapter不正确'
assert s['export_dtype']=='bfloat16' and s['tf32'] is False
assert s['tensors'] and any(t['delta_abs_max']>0 for t in s['tensors']), '真实A必须包含非零增量'
changed=[k for v in s['shards'].values() for k in v['changed_tensors']]
assert len(changed)==len(set(changed))==len(s['tensors'])
assert set(changed)=={t['key'] for t in s['tensors']}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert sha(Path(s['adapter'])/'adapter_model.safetensors')==s['adapter_sha256']
assert sha(Path(s['base'])/'model.safetensors.index.json')==s['base_index_sha256']
for shard,row in s['shards'].items():
    assert sha(p/shard)==row['merged_sha256'], shard
    assert sha(Path(s['base'])/shard)==row['base_sha256'], shard
print('merge产物完整，已核对分片和输入哈希；changed_tensors=',len(changed))
PYTHON

python3 /work/mol/evidence/check-merge.py /work/mol/merged-A /work/mol/adapter-A
