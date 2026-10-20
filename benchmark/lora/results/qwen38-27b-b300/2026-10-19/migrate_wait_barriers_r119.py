"""Replace only four verified childless waiters; never interrupt their GPU work."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
out=R/'wait-barrier-migration-r119.json';assert not out.exists()
items=[('pd',954267,'run_pd4_suite_r86.py'),('cached',959735,'run_cached_topologies_r90.py'),('isolated',959739,'run_isolated_speed_r91.py'),('million',1070664,'run_million_cache_r106.py')]
source=json.loads((R/'source-wait-barrier-recovery-r119.json').read_text())
for n,h in source['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert not Path('/proc/954883').exists()
assert json.loads((R/'dcp-candidate-probe-finished-r117.json').read_text())['passed']
assert not (R/'eight-gpu-suite-complete-r31.json').exists()
assert not (R/'pd4-precision-perf-complete-r33.json').exists()
for pattern in ['perf-pd4-*-r33','perf-cache-tp4-r50-*','perf-isolated-*-r69','million-cache-finished-r97.json']:
 assert not list(R.glob(pattern)),pattern
current=json.loads((R/'dcp-recovery-process-r118.json').read_text())
p=Path('/proc')/str(current['pid']);assert p.exists() and b'run_eight_gpu_recovery_r118.py' in (p/'cmdline').read_bytes()
assert not any(n.startswith('qwen38-pd4-') for n in subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines())
report={'at_unix':time.time(),'gpu_workers_interrupted':False,'original_failed_barrier_written':False,'new_barrier':'eight-gpu-recovery-complete-r118.json','source_manifest_sha256':hashlib.sha256((R/'source-wait-barrier-recovery-r119.json').read_bytes()).hexdigest(),'coordinators':[]}
def verify(pid,driver):
 p=Path(f'/proc/{pid}');cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
 assert cmd==f'python3 {R}/{driver}',cmd
 assert not (p/f'task/{pid}/children').read_text().strip()
 w=(p/'wchan').read_text();assert 'nanosleep' in w,w
 return {'old_pid':pid,'old_cmdline':cmd,'old_wchan':w,'old_children':[],'old_starttime_ticks':(p/'stat').read_text().split()[21]}
for kind,pid,driver in items:report['coordinators'].append({'kind':kind,**verify(pid,driver)})
# All four pass before any process is changed; each is checked again immediately.
for row,(kind,pid,driver) in zip(report['coordinators'],items):
 assert verify(pid,driver)['old_starttime_ticks']==row['old_starttime_ticks']
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  p=Path(f'/proc/{pid}')
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.05)
 else:raise RuntimeError('Old waiter did not stop')
 with (R/f'wait-barrier-{kind}-host-r119.log').open('x') as f:
  proc=subprocess.Popen(['python3',str(R/'resume_wait_barrier_r119.py'),'--kind',kind],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(1);assert proc.poll() is None,(kind,proc.poll())
 q=Path(f'/proc/{proc.pid}');assert not (q/f'task/{proc.pid}/children').read_text().strip()
 assert 'nanosleep' in (q/'wchan').read_text()
 row.update(new_pid=proc.pid,new_waiting_verified=True,new_cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),old_terminal=True)
 out.write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
