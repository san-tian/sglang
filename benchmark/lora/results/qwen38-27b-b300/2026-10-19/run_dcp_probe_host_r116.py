"""Preserve terminal failed cohort, build an isolated candidate, run real NCCL probe."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
W='/work/qwen38-mol-multigpu-20261019'
CID='cf207cd1072232e56fadd83a371e5123c3d606135f2faf0650c17df315a7f33a'
BASE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
def call(args):return subprocess.check_output(args,text=True).strip()
def save(name,data):
 with (R/name).open('x') as f:json.dump(data,f,indent=2)
assert not (R/'dcp-candidate-probe-finished-r116.json').exists()
manifest=json.loads((R/'source-runtime-candidate-r116.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/'source-r116'/n).read_bytes()).hexdigest()==h,n
assert call(['docker','image','inspect','qwen38-mol:r28','--format','{{.Id}}'])==BASE
assert shutil.disk_usage('/').free>4*1024**3
assert not Path('/proc/954883').exists() and not Path('/proc/1012683').exists()
scene=json.loads((R/'dcp8x2-failure-scene-r116.json').read_text())
audit=json.loads((R/'dcp8x2-mixed-failure-audit-r116.json').read_text())
assert scene['container_id']==CID and audit['passed'] is False
assert hashlib.sha256((R/'dcp8x2-failed-engine-r116.log').read_bytes()).hexdigest()==scene['failed_log_sha256']
assert json.loads(call(['docker','inspect',CID]))[0]['State']['Running']
save('dcp-candidate-probe-started-r116.json',{'started_at_unix':time.time(),'original_failure_preserved':True,'root_free_bytes':shutil.disk_usage('/').free})
# Only the failed coordinator's own healthy-but-stranded experiment is stopped.
call(['docker','stop',CID])
save('dcp-original-failed-cohort-stopped-r116.json',{'at_unix':time.time(),'container_id':CID,'state':json.loads(call(['docker','inspect',CID]))[0]['State']})
with (R/'image-build-r116.log').open('x') as f:
 subprocess.run(['docker','build','--pull=false','-t','qwen38-mol:r116',str(R/'source-r116')],stdout=f,stderr=subprocess.STDOUT,check=True)
image=call(['docker','image','inspect','qwen38-mol:r116','--format','{{.Id}}'])
assert image!=BASE
assert call(['docker','image','inspect','qwen38-mol:r28','--format','{{.Id}}'])==BASE
cache=R/'caches/dcp-kernel-r116';cache.mkdir(exist_ok=False,parents=True)
cmd=['docker','create','--name','qwen38-dcp-kernel-r116','--gpus','"device=0,1"','--network','host','--ipc','host','-v','/opt/dlami/nvme:/work','-v',str(cache)+':/root/.cache','-e','TMPDIR=/root/.cache/tmp','-e','PYTHONPYCACHEPREFIX=/root/.cache/pycache',image,'python3','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',W+'/probe_dcp_chunk_invariance_r116.py','--output',W+'/dcp-chunk-invariance-probe-r116.json','--original-kernel',W+'/extend_attention_r28_original_r116.py']
(cache/'tmp').mkdir()
save('dcp-candidate-image-r116.json',{'image_id':image,'base_image_id':BASE,'build_manifest_sha256':hashlib.sha256((R/'source-runtime-candidate-r116.json').read_bytes()).hexdigest(),'command':cmd})
call(cmd)
with (R/'dcp-chunk-invariance-probe-r116.log').open('x') as f:
 p=subprocess.run(['docker','start','-a','qwen38-dcp-kernel-r116'],stdout=f,stderr=subprocess.STDOUT)
state=json.loads(call(['docker','inspect','qwen38-dcp-kernel-r116']))[0]['State']
result=json.loads((R/'dcp-chunk-invariance-probe-r116.json').read_text()) if (R/'dcp-chunk-invariance-probe-r116.json').exists() else None
save('dcp-candidate-probe-finished-r116.json',{'finished_at_unix':time.time(),'returncode':p.returncode,'state':state,'image_id':image,'passed':p.returncode==0 and bool(result and result['passed']),'full_goal_complete':False})
print(json.dumps({'returncode':p.returncode,'result_present':result is not None,'passed':bool(result and result['passed'])}),flush=True)
raise SystemExit(p.returncode)
