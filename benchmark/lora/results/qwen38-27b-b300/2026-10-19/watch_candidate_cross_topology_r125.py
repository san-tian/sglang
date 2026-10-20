"""Wait for a sealed new merged reference; no GPU mutation on diagnostic failure."""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
source=json.loads((R/'source-candidate-cross-r125.json').read_text())
def verify():
 for n,h in source['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
verify();pid=json.loads((R/'dcp-recovery-process-r118.json').read_text())['pid']
result={'passed':False,'full_goal_complete':False,'scope':'Read-only candidate merged prefill cross-topology diagnostic, not D/M, decode or native speed acceptance.'};deadline=time.monotonic()+172800
try:
 while not (R/'dcp8x2-full-reference-r118/seal.json').exists():
  p=Path(f'/proc/{pid}');assert p.exists() and (p/'stat').read_text().split()[2]!='Z'
  assert b'run_eight_gpu_recovery_r118.py' in (p/'cmdline').read_bytes()
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; do not restart GPU workers')
  time.sleep(5)
 verify()
 with (R/'candidate-cross-prefill-r125.log').open('x') as f:
  ret=subprocess.run(['python3',str(R/'audit_candidate_cross_topology_r125.py'),'--root',str(R),'--output',str(R/'tp8-vs-dcp8x2-candidate-prefill-r125.json')],stdout=f,stderr=subprocess.STDOUT)
 report=json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())
 result.update(passed=report['passed'],returncode=ret.returncode,prefill=report['prefill'],report_sha256=hashlib.sha256((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_bytes()).hexdigest())
 assert ret.returncode==(0 if report['passed'] else 1)
except BaseException:result['error']=traceback.format_exc();raise
finally:
 result['at_unix']=time.time()
 with (R/'candidate-cross-prefill-finished-r125.json').open('x') as f:json.dump(result,f,indent=2)
