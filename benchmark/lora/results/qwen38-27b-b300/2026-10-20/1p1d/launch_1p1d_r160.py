"""Two physical B300 GPUs, TP1 P + TP1 D, existing validated R55 runtime."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
W='/work/qwen38-mol-multigpu-20261019'
IMAGE='sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
def main():
 reservation=json.loads((R/'priority-1p1d-reservation-r154.json').read_text())
 assert reservation['gpu_apps_after'].strip()==''
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
 names=[]
 for role,gpu,port,nccl in [('prefill',0,31800,31820),('decode',1,31801,31821)]:
  name=f'qwen38-1p1d-{role}-r160'; names.append(name)
  cache=R/'caches'/name
  shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache)
  (cache/'tmp').mkdir(parents=True,exist_ok=True)
  tune=cache/'sglang/flashinfer/autotune/0.7.2/sm103/add08325a8bdc6e8/rank_tp0_pp0_dp0.json'
  tune.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile('/opt/dlami/nvme/qwen38-mol-fork-20261019/autotune-valid-trace-v2.json',tune)
  env={'PYTHONPATH':W+'/client-r33','TMPDIR':'/root/.cache/tmp','PYTHONPYCACHEPREFIX':'/root/.cache/pycache','SGLANG_DISAGGREGATION_NIXL_BACKEND':'UCX','UCX_TLS':'cuda_copy,cuda_ipc,posix,sysv,tcp','UCX_LOG_LEVEL':'info'}
  cmd=['docker','create','--name',name,'--gpus','"device=0,1"','--network','host','--ipc','host','--pid','host','-v','/opt/dlami/nvme:/work','-v',str(cache)+':/root/.cache']
  for k,v in env.items():cmd+=['-e',k+'='+v]
  cmd += [IMAGE,'python3','-m','sglang.launch_server','--model-path','/work/models/Qwen3.8-27B','--served-model-name','qwen38-27b-1p1d','--host','127.0.0.1','--port',str(port),'--tp-size','1','--context-length','262144','--mem-fraction-static','0.80','--max-mamba-cache-size','256','--max-running-requests','64','--chunked-prefill-size','8192','--disable-cuda-graph','--enable-custom-logit-processor','--linear-attn-prefill-backend','triton','--enable-deterministic-inference','--attention-backend','triton','--bf16-gemm-backend','torch','--enable-lora','--max-lora-rank','32','--max-loras-per-batch','16','--max-loaded-loras','32','--lora-target-modules','qkv_proj','o_proj','gate_up_proj','down_proj','in_proj_qkvz','in_proj_ba','out_proj','--lora-backend','merged_bf16','--lora-merged-bf16-cache-size-mb','36864','--lora-merged-bf16-cache-policy','active','--random-seed','0','--base-gpu-id',str(gpu),'--disaggregation-mode',role,'--disaggregation-bootstrap-port','31810','--nccl-port',str(nccl),'--disaggregation-transfer-backend','nixl','--mamba-radix-cache-strategy','extra_buffer','--enable-metrics','--watchdog-timeout','900']
  with (R/f'launch-{name}.json').open('x') as f:json.dump(cmd,f,indent=2)
  subprocess.run(cmd,check=True)
 # Own and create both roles before starting either role.
 subprocess.run(['docker','start',*names],check=True)
 with (R/'1p1d-launch-r160.json').open('x') as f:json.dump({'unix_s':time.time(),'runtime_image':IMAGE,'names':names,'gpus':{'prefill':[0],'decode':[1]},'context_length':262144,'max_loras_per_batch':16,'max_loaded_loras':32,'cache_MiB_per_role':36864,'status':'starting; not performance acceptance'},f,indent=2)
if __name__=='__main__':main()
