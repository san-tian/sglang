"""Wait for the actual PD coordinator; audit full precision only after success."""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
source=json.loads((R/'source-audit-r94.json').read_text())
for n,h in source['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
marker=R/'pd4-precision-perf-complete-r33.json';pid=954267;end=time.monotonic()+172800
result={'passed':False,'full_goal_complete':False,'scope':'Full4P4D original main/GDN score audit only, not native timing/kernel/capacity acceptance.'}
try:
 while not marker.exists():
  proc=Path(f'/proc/{pid}')
  if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':
   raise RuntimeError('Authoritative PD coordinator terminal without success marker; preserve failure, do not accept or restart')
  assert b'run_pd4_suite_r86.py' in (proc/'cmdline').read_bytes()
  if time.monotonic()>end:raise TimeoutError('Audit observation window expired; do not restart GPU workers')
  time.sleep(5)
 assert json.loads(marker.read_text())['natural_outputs_equal']
 with (R/'pd4-original-precision-audit-r94.log').open('x') as f:
  subprocess.run(['python3',str(R/'audit_pd_precision_r94.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 report=json.loads((R/'pd4-original-precision-audit-r94.json').read_text());assert report['passed']
 result.update(passed=True,audit='pd4-original-precision-audit-r94.json')
except BaseException:
 result['error']=traceback.format_exc();raise
finally:
 result['unix_s']=time.time()
 with (R/'pd-precision-audit-finished-r94.json').open('x') as f:json.dump(result,f,indent=2)
