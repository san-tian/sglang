import hashlib,json,subprocess
from pathlib import Path
r=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
assert not (r/'eight-gpu-suite-started-r31.json').exists()
assert not (r/'active-cache-candidate-finished-r56.json').exists()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
usage={int(a):[int(b),int(c)] for a,b,c in [x.split(',') for x in rows.splitlines()]}
assert all(usage[i][0]<2000 and usage[i][1]==0 for i in (0,5)),usage
(r/'nixl-two-process-r74').mkdir(exist_ok=False)
image='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288';names=[];commands={};processes=[]
for role,gpu in [('destination',5),('source',0)]:
 name='qwen38-nixl-'+role+'-r74';names.append(name)
 cmd=['docker','run','--name',name,'--gpus','"device='+str(gpu)+'"','--ipc','host','--network','host','--entrypoint','python3','-e','UCX_TLS=cuda_copy,cuda_ipc,posix,sysv,tcp','-e','UCX_LOG_LEVEL=info','-v',str(r)+':/evidence',image,'/evidence/preflight_nixl_process_r74.py',role];commands[role]=cmd
 f=(r/(name+'.log')).open('x');processes.append((subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT),f))
(r/'nixl-two-process-launch-r74.json').write_text(json.dumps({'before_gpu_usage':usage,'commands':commands},indent=2))
for proc,f in processes:proc.wait();f.close()
proof={'before_gpu_usage':usage,'commands':commands,'containers':{}}
for name in names:
 x=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
 proof['containers'][name]={'actual_image':x['Image'],'actual_devices':x['HostConfig']['DeviceRequests'],'state':x['State']}
(r/'nixl-two-process-launch-r74.json').write_text(json.dumps(proof,indent=2))
assert all(proc.returncode==0 for proc,f in processes),[proc.returncode for proc,f in processes]
