"""Serialize measured loads; keep completed and failed cohort boundaries explicit."""
import json,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'fp8-mtp-r228'
def wait(kind,phase):
 p=E/kind/phase
 for i in range(900):
  if (p/'failed.json').exists():raise RuntimeError((kind,phase,(p/'failed.json').read_text()))
  if (p/'complete.json').exists():return
  time.sleep(1)
 raise TimeoutError((kind,phase))
def run(kind,phase):
 print('START',kind,phase,flush=True)
 subprocess.run(['python3','-u',str(R/'measure_fp8_r232.py'),kind,phase],check=True)
 print('DONE',kind,phase,flush=True)
wait('pd','speed');wait('colocated','smoke');run('colocated','speed')
wait('off','smoke');run('off','speed')
for kind in ['pd','colocated']:run(kind,'capacity')
for kind in ['pd','colocated']:run(kind,'slots')
(E/'suite-complete-r233.json').write_text(json.dumps({'unix_s':time.time(),'scope':'User FP8/MTP smoke, speed, capacity, adapter registration/eviction; independent quantized-merge precision not certified.'},indent=2))
