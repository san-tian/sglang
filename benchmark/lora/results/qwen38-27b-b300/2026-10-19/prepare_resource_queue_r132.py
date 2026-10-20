"""Release independent experiments on a resource lease, never on fake accuracy success."""
import ast,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(name):return hashlib.sha256((R/name).read_bytes()).hexdigest()
def save(name,text):
 p=R/name;assert not p.exists();ast.parse(text);p.write_text(text)

wrapper=(R/'resume_wait_barrier_r119.py').read_text()
wrapper=wrapper.replace('source-wait-barrier-recovery-r119.json','source-resource-queue-r132.json')
wrapper=wrapper.replace("NEW='eight-gpu-recovery-complete-r118.json'", "NEW='attention-dp8-resource-released-r131.json'")
wrapper=wrapper.replace('effective-wait-barriers-r119.json','effective-wait-barriers-r132.json')
start=wrapper.index("  cohort=json.loads((R/NEW).read_text())")
end=wrapper.index(" module.waitleases=wait",start)
wrapper=wrapper[:start]+'''  resource=json.loads((R/NEW).read_text())
  assert resource['resource_released'] and resource['observed_gpu_processes']==''
  assert resource['model_accuracy_accepted'] is False and resource['full_goal_complete'] is False
  process=json.loads((R/'attention-dp8-process-r131.json').read_text())
  assert process['source_sha256']==hashlib.sha256((R/'run_attention_dp8_r131.py').read_bytes()).hexdigest()
  # The lease record is the producer's last action; wait for its actual exit.
  proc=Path('/proc')/str(process['pid'])
  while proc.exists() and (proc/'stat').read_text().split()[2]!='Z':
   assert b'run_attention_dp8_r131.py' in (proc/'cmdline').read_bytes()
   time.sleep(.2)
  import subprocess
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
  assert json.loads((R/'tp8-original-audits-finished-r102.json').read_text())['passed']
  assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False
  assert not (R/OLD).exists() and not (R/'eight-gpu-recovery-complete-r118.json').exists()
'''+wrapper[end:]
save('resume_resource_lease_r132.py',wrapper)

migration=(R/'migrate_wait_barriers_r123.py').read_text()
migration=migration.replace('wait-barrier-migration-r123.json','resource-queue-migration-r132.json')
a=migration.index('items=[');b=migration.index("source=json.loads",a)
migration=migration[:a]+'''previous=json.loads((R/'wait-barrier-migration-r123.json').read_text())
items=[(row['kind'],row['new_pid'],'resume_wait_barrier_r119.py') for row in previous['coordinators']]
assert len(items)==4 and {x[0] for x in items}=={'pd','cached','isolated','million'}
'''+migration[b:]
migration=migration.replace('source-wait-barrier-recovery-r123.json','source-resource-queue-r132.json')
migration=migration.replace("assert json.loads((R/'dcp-candidate-probe-finished-r117.json').read_text())['passed']", "assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False\nassert not (R/'eight-gpu-recovery-complete-r118.json').exists()")
migration=migration.replace('dcp-recovery-process-r118.json','attention-dp8-process-r131.json').replace('run_eight_gpu_recovery_r118.py','run_attention_dp8_r131.py')
migration=migration.replace("'new_barrier':'eight-gpu-recovery-complete-r118.json'", "'new_barrier':'attention-dp8-resource-released-r131.json','barrier_semantics':'GPU resource release only, never model correctness or performance acceptance'")
migration=migration.replace('def verify(pid,driver):','def verify(pid,driver,kind):')
migration=migration.replace("expected=f'python3 -u {R}/{driver}' if driver=='run_million_cache_r106.py' else f'python3 {R}/{driver}'", "expected=f'python3 {R}/{driver} --kind {kind}'")
migration=migration.replace('verify(pid,driver)','verify(pid,driver,kind)')
migration=migration.replace('wait-barrier-{kind}-host-r119.log','resource-queue-{kind}-host-r132.log')
migration=migration.replace("R/'resume_wait_barrier_r119.py'","R/'resume_resource_lease_r132.py'")
save('migrate_resource_queue_r132.py',migration)
base=json.loads((R/'source-wait-barrier-recovery-r123.json').read_text())['source_files']
files=set(base)|{'resume_resource_lease_r132.py','migrate_resource_queue_r132.py','source-host-r131.json','run_attention_dp8_r131.py','wait-barrier-migration-r123.json'}
record={'source_files':{n:sha(n) for n in sorted(files)},
        'scope':'Four childless CPU waiters migrate from impossible numerical-success barrier to an explicit experiment GPU-resource-release barrier. Original experiment, runtime, clients and measurement sources are unchanged. All original failed/invalid completion evidence remains preserved; no accuracy or performance acceptance is inferred.'}
with (R/'source-resource-queue-r132.json').open('x') as f:json.dump(record,f,indent=2)

watch=(R/'watch_pd_after_migration_r124.py').read_text().replace('r124','r133')
watch=watch.replace('wait-barrier-migration-r123.json','resource-queue-migration-r132.json')
watch=watch.replace('resume_wait_barrier_r119.py','resume_resource_lease_r132.py')
save('watch_pd_after_resource_queue_r133.py',watch)
base=json.loads((R/'source-pd-after-migration-r124.json').read_text())['source_files']
files=set(base)|{'watch_pd_after_resource_queue_r133.py','resume_resource_lease_r132.py','source-resource-queue-r132.json'}
files.discard('watch_pd_after_migration_r124.py')
record={'source_files':{n:sha(n) for n in sorted(files)},'scope':'Read-only watch of the actual R132 replacement PD coordinator; unchanged original R94/R95 raw auditors, thresholds and counts.'}
with (R/'source-pd-after-migration-r133.json').open('x') as f:json.dump(record,f,indent=2)
print('Prepared resource-only queue migration and unchanged original PD audit continuation')
