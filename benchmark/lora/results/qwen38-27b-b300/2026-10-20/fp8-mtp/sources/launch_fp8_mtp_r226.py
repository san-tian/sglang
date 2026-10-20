"""User-priority FP8+FP8KV+MTP, isolated two-card PD, no CP/DPA flags."""
import json,shutil,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'fp8-mtp-r226'
model=Path('/opt/dlami/nvme/models/Qwen3.8-27B-FP8');assert len(list(model.glob('*.safetensors')))==66
used={int(a):int(b) for a,b in [s.split(',') for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]};assert all(used[g]<2000 for g in [2,3]),used
for role,base,port in [('prefill',0,33800),('decode',1,34800)]:
 cmd=json.loads((R/f'launch-qwen38-1p1d-{role}-r160.json').read_text());name=f'qwen38-fp8-mtp-{role}-r226'
 assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
 def change(flag,v):cmd[cmd.index(flag)+1]=str(v)
 def drop(flag,argument=True):
  if flag in cmd:
   i=cmd.index(flag);del cmd[i:i+2 if argument else i+1]
 for flag in ['--disable-cuda-graph','--enable-deterministic-inference','--enable-custom-logit-processor']:drop(flag,False)
 for flag in ['--max-mamba-cache-size','--attention-backend','--bf16-gemm-backend','--linear-attn-prefill-backend','--lora-merged-bf16-cache-size-mb','--lora-merged-bf16-cache-policy']:drop(flag)
 change('--name',name);change('--gpus','"device=2,3"');change('--base-gpu-id',base);change('--port',port);change('--nccl-port',port+20);change('--disaggregation-bootstrap-port',33810);change('--model-path','/work/models/Qwen3.8-27B-FP8');change('--served-model-name','qwen38-fp8-mtp-r226');change('--mamba-radix-cache-strategy','extra_buffer_lazy');change('--lora-backend','triton')
 cache=E/'caches'/name
 def skip(d,ns):return [x for x in ns if not (Path(d)/x).is_file() and not (Path(d)/x).is_dir() and not (Path(d)/x).is_symlink()]
 shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache,symlinks=True,ignore=skip);(cache/'tmp').mkdir(exist_ok=True)
 for i,v in enumerate(cmd):
  if v.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
 cmd+=['--kv-cache-dtype','fp8_e4m3','--reasoning-parser','qwen3','--tool-call-parser','qwen3_coder','--mamba-full-memory-ratio','3.74','--speculative-algorithm','EAGLE','--speculative-num-steps','3','--speculative-eagle-topk','1','--speculative-num-draft-tokens','4','--mamba-ssm-dtype','bfloat16']
 (E/('launch-'+name+'.json')).write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True)
subprocess.run(['docker','start','qwen38-fp8-mtp-prefill-r226','qwen38-fp8-mtp-decode-r226'],check=True)
(E/'started.json').write_text(json.dumps({'unix_s':time.time(),'physical_gpus':{'prefill':2,'decode':3},'primary_r160_preserved':True,'note':'Native FP8 compute and CUDA graphs use backend defaults; removed BF16 experiment deterministic/eager/GEMM/cache flags, and explicit Mamba-size pin so requested ratio can apply.'},indent=2))
