"""Run independent FP64 arithmetic after releasing the instrumented server."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

R=Path('/work/qwen38-mol-multigpu-20261019')
root=R/'real-qkv-r129'
assert json.loads((root/'capture-finished.json').read_text())['diagnostic_captured']
assert not (root/'oracle-finished.json').exists()
workers=[]
for rank in range(2):
    log=(root/f'oracle-rank{rank}.log').open('x')
    worker=subprocess.Popen([sys.executable,str(R/'oracle_real_qkv_r129.py'),
        '--root',str(root),'--rank',str(rank),'--world','2'],stdout=log,stderr=subprocess.STDOUT)
    workers.append((worker,log))
returns=[worker.wait() for worker,log in workers]
for worker,log in workers:log.close()
assert returns==[0,0],returns
import torch
torch.set_num_threads(1)
files=sorted(root.glob('*-rank*-layer*.pt'))
records=[row for rank in range(2) for row in json.loads((root/f'oracle-rank{rank}.json').read_text())['records']]
assert len(files)==len(records)==48
assert {row['file'] for row in records}=={p.name for p in files}
replicas=[]
for p in files:
    parts=p.stem.split('-')
    rank=int(parts[1][4:])
    if rank%2:
        continue
    peer=root/(p.name.replace(f'-rank{rank}-',f'-rank{rank+1}-'))
    a=torch.load(p,map_location='cpu',weights_only=True)
    b=torch.load(peer,map_location='cpu',weights_only=True)
    assert a['lengths']==b['lengths']
    for key in ['k','v']:
        assert torch.equal(a[key],b[key]),(p.name,key)
    replicas.append({'left':p.name,'right':peer.name,'replicated_KV_bit_exact':True})
result={'diagnosis_completed':True,'model_accuracy_accepted':False,'numerical_failure_fixed':False,
        'records':records,'replicated_head_pairs':replicas,
        'all_KV_write_readbacks_exact':all(x['k_pool_exact'] and x['v_pool_exact'] for x in records),
        'server_recorded_means_valid':False,
        'statistics_note':'Server deterministic dispatcher produced some recorded means larger than their maxima. Those means are invalid and are replaced only analytically by this independent FP64 re-computation; original captures remain unchanged.',
        'sources_sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['oracle_real_qkv_r129.py','run_real_qkv_oracle_r129.py']},
        'full_goal_complete':False,
        'scope':'Real model first two full-attention layers for short, long and packed-batch prefixes; exact K/V write read-back, replicated head equality and FP64 attention-only oracles. No full-model acceptance or speed claim.'}
with (root/'oracle-finished.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['diagnosis_completed','all_KV_write_readbacks_exact','numerical_failure_fixed','server_recorded_means_valid']}))
