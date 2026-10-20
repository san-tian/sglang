import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
W='/work/qwen38-mol-multigpu-20261019'
name='qwen38-real-qkv-oracle-r129'
assert json.loads((R/'real-qkv-r129/capture-finished.json').read_text())['diagnostic_captured']
assert not json.loads(subprocess.check_output(['docker','inspect','qwen38-dcp8x2-qkv-probe-r129']))[0]['State']['Running']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cache=R/'caches'/name;cache.mkdir(parents=True,exist_ok=False);(cache/'tmp').mkdir()
cmd=['docker','create','--name',name,'--gpus','"device=0,1"','--network','host','--ipc','host',
     '-v','/opt/dlami/nvme:/work','-v',str(cache)+':/root/.cache',
     '-e','TMPDIR=/root/.cache/tmp','-e','PYTHONPYCACHEPREFIX=/root/.cache/pycache',
     'sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878',
     'python3',W+'/run_real_qkv_oracle_r129.py']
subprocess.run(cmd,check=True)
with (R/'real-qkv-oracle-host-r129.log').open('x') as f:
    result=subprocess.run(['docker','start','-a',name],stdout=f,stderr=subprocess.STDOUT)
info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
with (R/'real-qkv-oracle-state-r129.json').open('x') as f:
    json.dump({'returncode':result.returncode,'state':info['State'],'container_id':info['Id'],
               'image':info['Image'],'command':cmd,'at_unix':time.time(),
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f,indent=2)
raise SystemExit(result.returncode)
