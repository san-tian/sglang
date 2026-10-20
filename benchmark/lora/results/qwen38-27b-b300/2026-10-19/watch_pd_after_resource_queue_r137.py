"""Watch the actual migrated PD coordinator and run unchanged original raw audits."""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
source=json.loads((R/'source-pd-after-migration-r137.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 for n,h in source['source_files'].items():assert sha(R/n)==h,n
verify()
migration=json.loads((R/'resource-queue-migration-r136.json').read_text())
row=next(x for x in migration['coordinators'] if x['kind']=='pd');pid=row['new_pid']
marker=R/'pd4-precision-perf-complete-r33.json';deadline=time.monotonic()+172800
result={'passed':False,'full_goal_complete':False,'coordinator_pid':pid,'migration_sha256':sha(R/'resource-queue-migration-r136.json'),'scope':'Original full4P4D R94 precision and R95 natural/kernel audits, unchanged; monitor only the verified migrated CPU coordinator. Original old-PID observers and any terminal reports preserved.'}
try:
 while not marker.exists():
  p=Path(f'/proc/{pid}');assert p.exists() and (p/'stat').read_text().split()[2]!='Z','Actual migrated PD coordinator terminal without completion evidence'
  cmd=(p/'cmdline').read_bytes();assert b'resume_resource_lease_r136.py' in cmd and b'--kind\x00pd' in cmd
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; do not restart GPU workers')
  time.sleep(5)
 assert json.loads(marker.read_text())['natural_outputs_equal']
 for rev,kind in [(94,'precision'),(95,'native')]:
  verify();output=R/('pd4-original-precision-audit-r94.json' if kind=='precision' else 'pd4-native-kernel-audit-r95.json');assert not output.exists()
  with (R/f'pd4-{kind}-after-migration-r137.log').open('x') as f:
   subprocess.run(['python3',str(R/f'audit_pd_{kind}_r{rev}.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
  report=json.loads(output.read_text());assert report['passed']
  result[kind]={'report':output.name,'sha256':sha(output)}
 result['passed']=True
except BaseException:result['error']=traceback.format_exc();raise
finally:
 result['at_unix']=time.time()
 with (R/'pd-original-audits-after-migration-r137.json').open('x') as f:json.dump(result,f,indent=2)
