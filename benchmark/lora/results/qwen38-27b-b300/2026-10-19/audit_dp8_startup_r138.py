"""Bind actual effective per-worker SSM slots and scheduler GPU placement."""
import hashlib,json,re,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
raw=subprocess.check_output(['docker','logs','qwen38-dpa8-m-real_r16-r135'],stderr=subprocess.STDOUT,text=True)
pattern=re.compile(r'DP(\d+) TP(\d+)\].*Mamba Cache is allocated\. max_mamba_cache_size: (\d+), conv_state size: ([0-9.]+)GB, ssm_state size: ([0-9.]+)GB')
alloc={}
for line in raw.splitlines():
    m=pattern.search(line)
    if m:
        alloc[int(m[1])]={'tp_rank':int(m[2]),'slots':int(m[3]),
                         'conv_GiB_rounded':float(m[4]),'SSM_GiB_rounded':float(m[5]),'log_line':line}
assert set(alloc)==set(range(8)) and all(x['slots']==32 for x in alloc.values())
config=json.loads((R/'attention-dp8-first-config-r138.json').read_text())
assert config['max_mamba_cache_size']==256 and config['attn_dp_size']==8
states=config['internal_states']
assert len(states)==8 and all(x['max_mamba_cache_size']==32 and x['effective_max_running_requests_per_dp']==4 for x in states)
placement=json.loads((R/'attention-dp8-first-placement-r138.json').read_text())
assert placement['placement_matched'] and len(placement['scheduler_gpu_processes'])==8
report={'startup_passed':True,'model_accuracy_accepted':False,'native_speed_accepted':False,
        'equal_byte_KV_accepted':False,'CLI_global_SSM_slots':256,'per_worker_SSM_slots':32,
        'effective_max_requests_per_DP':4,'allocations':alloc,
        'actual_GPU_mapping':placement['scheduler_gpu_processes'],
        'config_sha256':hashlib.sha256((R/'attention-dp8-first-config-r138.json').read_bytes()).hexdigest(),
        'placement_sha256':hashlib.sha256((R/'attention-dp8-first-placement-r138.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'at_unix':time.time(),'full_goal_complete':False}
with (R/'attention-dp8-startup-proof-r138.json').open('x') as f:json.dump(report,f,indent=2)
print('Verified8 actual SSM allocations,8 effective limits and8 GPU placements')
