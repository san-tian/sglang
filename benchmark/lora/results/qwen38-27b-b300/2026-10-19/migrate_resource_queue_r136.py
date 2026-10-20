"""Replace only four verified childless waiters; never interrupt their GPU work."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
out=R/'resource-queue-migration-r136.json';assert not out.exists()
previous=json.loads((R/'wait-barrier-migration-r123.json').read_text())
items=[(row['kind'],row['new_pid'],'resume_wait_barrier_r119.py') for row in previous['coordinators']]
assert len(items)==4 and {x[0] for x in items}=={'pd','cached','isolated','million'}
source=json.loads((R/'source-resource-queue-r136.json').read_text())
for n,h in source['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert not Path('/proc/954883').exists()
assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False
assert not (R/'eight-gpu-recovery-complete-r118.json').exists()
assert not (R/'eight-gpu-suite-complete-r31.json').exists()
assert not (R/'pd4-precision-perf-complete-r33.json').exists()
for pattern in ['perf-pd4-*-r33','perf-cache-tp4-r50-*','perf-isolated-*-r69','million-cache-finished-r97.json']:
 assert not list(R.glob(pattern)),pattern
current=json.loads((R/'attention-dp8-process-r135.json').read_text())
p=Path('/proc')/str(current['pid'])
if p.exists() and (p/'stat').read_text().split()[2]!='Z':
 assert b'run_attention_dp8_r135.py' in (p/'cmdline').read_bytes()
else:
 assert json.loads((R/'attention-dp8-resource-released-r135.json').read_text())['resource_released']
assert not any(n.startswith('qwen38-pd4-') for n in subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines())
report={'at_unix':time.time(),'gpu_workers_interrupted':False,'original_failed_barrier_written':False,'new_barrier':'attention-dp8-resource-released-r135.json','barrier_semantics':'GPU resource release only, never model correctness or performance acceptance','source_manifest_sha256':hashlib.sha256((R/'source-resource-queue-r136.json').read_bytes()).hexdigest(),'coordinators':[]}
def verify(pid,driver,kind):
 p=Path(f'/proc/{pid}');cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
 expected=f'python3 {R}/{driver} --kind {kind}'
 assert cmd==expected,cmd
 assert not (p/f'task/{pid}/children').read_text().strip()
 w=(p/'wchan').read_text();assert 'nanosleep' in w,w
 return {'old_pid':pid,'old_cmdline':cmd,'old_wchan':w,'old_children':[],'old_starttime_ticks':(p/'stat').read_text().split()[21]}
for kind,pid,driver in items:report['coordinators'].append({'kind':kind,**verify(pid,driver,kind)})
# All four pass before any process is changed; each is checked again immediately.
for row,(kind,pid,driver) in zip(report['coordinators'],items):
 assert verify(pid,driver,kind)['old_starttime_ticks']==row['old_starttime_ticks']
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  p=Path(f'/proc/{pid}')
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.05)
 else:raise RuntimeError('Old waiter did not stop')
 with (R/f'resource-queue-{kind}-host-r136.log').open('x') as f:
  proc=subprocess.Popen(['python3',str(R/'resume_resource_lease_r136.py'),'--kind',kind],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(1);assert proc.poll() is None,(kind,proc.poll())
 q=Path(f'/proc/{proc.pid}');assert not (q/f'task/{proc.pid}/children').read_text().strip()
 assert 'nanosleep' in (q/'wchan').read_text()
 row.update(new_pid=proc.pid,new_waiting_verified=True,new_cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),old_terminal=True)
 out.write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
