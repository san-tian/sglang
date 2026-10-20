"""Switch the dedicated eight-GPU experiment host to the authorized 4P4D baseline."""
import concurrent.futures as cf
import json, os, signal, shutil, subprocess, time
from pathlib import Path

R = Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
E = R / 'pd4-baseline-r253'
E.mkdir(exist_ok=False)
OLD = ['qwen38-1p1d-prefill-r160', 'qwen38-1p1d-decode-r160',
       'qwen38-fp8-mtp-colocated-r228', 'qwen38-fp8-off-colocated-r230',
       'qwen38-fp8-mtp-prefill-r228', 'qwen38-fp8-mtp-decode-r228']
def save(n, v): (E/n).write_text(json.dumps(v, indent=2))
save('switch-before.json', {'unix_s':time.time(), 'stopped_experiments':OLD,
    'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),
    'restore_argv':['docker','start']+OLD})
save('old-containers.json',json.loads(subprocess.check_output(['docker','inspect']+OLD,text=True)))
# Cancel only identified legacy measurement/lease scripts; they must not reclaim GPUs.
cancelled=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try: argv=(p/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if any(x.decode(errors='replace') in [str(R/n) for n in ['resume_resource_lease_r136.py','measure_1p1d_r154.py','measure_1p1d_r156.py']] for x in argv):
  os.kill(int(p.name),signal.SIGTERM);cancelled.append({'pid':int(p.name),'argv':[x.decode(errors='replace') for x in argv]})
save('cancelled-legacy-queues.json',cancelled)
subprocess.run(['docker','stop','--time','30']+OLD,check=True)
for _ in range(60):
 used=[int(s) for s in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
 if all(v<2000 for v in used): break
 time.sleep(1)
else: raise RuntimeError(('GPU memory not released',used))
workers=[]
for gpu in range(8):
 role='prefill' if gpu<4 else 'decode'; index=gpu%4
 port=40000+gpu*100; bootstrap=41000+index*100
 name=f'qwen38-pd4-{role}-{index}-r253'
 cmd=json.loads((R/'fp8-mtp-r228'/f'launch-qwen38-fp8-mtp-{role}-r228.json').read_text())
 def change(flag,value):cmd[cmd.index(flag)+1]=str(value)
 change('--name',name);change('--gpus','"device=0,1,2,3,4,5,6,7"')
 change('--base-gpu-id',gpu);change('--port',port);change('--nccl-port',42000+gpu*100)
 change('--disaggregation-bootstrap-port',bootstrap);change('--served-model-name','qwen38-pd4-baseline')
 cache=E/'caches'/name
 def skip(d,ns):return [x for x in ns if not (Path(d)/x).is_file() and not (Path(d)/x).is_dir() and not (Path(d)/x).is_symlink()]
 shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache,symlinks=True,ignore=skip)
 (cache/'tmp').mkdir(exist_ok=True)
 for i,v in enumerate(cmd):
  if v.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
 save('launch-'+name+'.json',cmd)
 subprocess.run(cmd,check=True)
 workers.append({'name':name,'role':role,'index':index,'physical_gpu':gpu,'url':f'http://127.0.0.1:{port}','bootstrap_port':bootstrap})
save('topology.json',{'workers':workers,'tp_size':1,'prefill_replicas':4,'decode_replicas':4,
 'dispatch':'client-side balanced pairing; P and D chosen independently; no production routing changes',
 'model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','runtime_overlay':'0116d3a12bc893ae43d8dbfd594280eccdc8467c',
 'image':'sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'})
subprocess.run(['docker','start']+[w['name'] for w in workers],check=True)
save('started.json',{'unix_s':time.time(),'workers':workers,'ready':False})
print('ALL_EIGHT_STARTED',flush=True)
