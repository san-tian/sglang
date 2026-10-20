"""Requested FP8/MTP exact colocated case and documented PD-compatible case."""
import json,shutil,subprocess,time,sys
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'fp8-mtp-r228';E.mkdir(exist_ok=True)
kind=sys.argv[1];assert kind in ['colocated','pd']
roles=[('colocated',2,0,35800)] if kind=='colocated' else [('prefill',4,0,36800),('decode',5,1,37800)]
used={int(a):int(b) for a,b in [s.split(',') for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]};assert all(used[g]<2000 for _,g,_,_ in roles),used
names=[]
for role,gpu,base,port in roles:
 original='decode' if role=='decode' else 'prefill'
 cmd=json.loads((R/'fp8-mtp-r226'/f'launch-qwen38-fp8-mtp-{original}-r226.json').read_text());name=f'qwen38-fp8-mtp-{role}-r228';names.append(name)
 assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
 def change(flag,v):cmd[cmd.index(flag)+1]=str(v)
 def drop(flag):
  if flag in cmd:i=cmd.index(flag);del cmd[i:i+2]
 change('--name',name);change('--gpus','device=2' if kind=='colocated' else '"device=4,5"');change('--base-gpu-id',base);change('--port',port);change('--nccl-port',port+20);change('--served-model-name',f'qwen38-fp8-mtp-{kind}-r228')
 if kind=='colocated':
  for flag in ['--disaggregation-mode','--disaggregation-bootstrap-port','--disaggregation-transfer-backend']:drop(flag)
 else:
  change('--disaggregation-bootstrap-port',36810);change('--mamba-radix-cache-strategy','extra_buffer')
 cache=E/'caches'/name
 def skip(d,ns):return [x for x in ns if not (Path(d)/x).is_file() and not (Path(d)/x).is_dir() and not (Path(d)/x).is_symlink()]
 shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache,symlinks=True,ignore=skip);(cache/'tmp').mkdir(exist_ok=True)
 for i,v in enumerate(cmd):
  if v.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
 (E/('launch-'+name+'.json')).write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True)
subprocess.run(['docker','start']+names,check=True)
(E/f'started-{kind}.json').write_text(json.dumps({'unix_s':time.time(),'physical_gpus':{r:g for r,g,b,p in roles},'primary_r160_preserved':True,'deviation':'none for user flags' if kind=='colocated' else 'PD requires extra_buffer instead of extra_buffer_lazy'},indent=2))
