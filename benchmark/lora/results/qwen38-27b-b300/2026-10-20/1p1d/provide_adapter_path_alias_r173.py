"""Expose the host-absolute symlink targets of synthetic capacity adapters."""
import json,subprocess,time,traceback
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
NAMES=['qwen38-1p1d-slots32-prefill-r169','qwen38-1p1d-slots32-decode-r169']
CODE="from pathlib import Path; Path('/opt/dlami').mkdir(parents=True,exist_ok=True); p=Path('/opt/dlami/nvme'); p.symlink_to('/work',target_is_directory=True) if not p.exists() else None; assert p.resolve()==Path('/work'); assert (p/'qwen38-mol-experiment/adapters/real_r16/adapter_model.safetensors').is_file(); print('absolute adapter target resolves to mounted /work')"
deadline=time.monotonic()+7200
pending=set(NAMES)
while pending:
 for name in sorted(pending):
  result=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
  if result.returncode:continue
  item=json.loads(result.stdout)[0]
  assert item['Image']=='sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
  if not item['State']['Running']:continue
  cmd=['docker','exec',name,'python3','-c',CODE]
  output=subprocess.check_output(cmd,text=True)
  with (R/f'adapter-path-alias-{name}-r173.json').open('x') as f:json.dump({'unix_s':time.time(),'container_id':item['Id'],'command':cmd,'output':output,'scope':'Container path alias only; no runtime, weights, sampling or timing code changed'},f,indent=2)
  pending.remove(name)
 if time.monotonic()>deadline:raise TimeoutError(pending)
 if pending:time.sleep(2)
