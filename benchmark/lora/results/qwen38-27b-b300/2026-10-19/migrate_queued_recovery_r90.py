import hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
out=R/'queued-recovery-migration-r90.json';assert not out.exists()
items=[(781207,'run_cached_topologies_r50.py','run_cached_topologies_r90.py',90),(903028,'run_isolated_speed_r69.py','run_isolated_speed_r91.py',91)]
assert not (R/'pd4-precision-perf-complete-r33.json').exists()
assert not (R/'cached-topologies-finished-r50.json').exists()
assert not (R/'isolated-speed-finished-r69.json').exists()
assert not list(R.glob('perf-cache-tp4-r50-*')) and not list(R.glob('perf-isolated-*-r69'))
names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
assert not any(n.startswith(('qwen38-cache-tp4-r50','qwen38-cache-dpa4-r50','qwen38-isolated-')) for n in names)
proof={'unix_s':time.time(),'runtime_workers_interrupted':False,'scope':'Only two verified waiting, childless CPU coordinators replaced. Full native counts, topology controls, audits and completion markers unchanged.','coordinators':[]}
for pid,old,new,rev in items:
 m=json.loads((R/f'source-host-r{rev}.json').read_text())
 for n,h in m['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 p=Path(f'/proc/{pid}');cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
 assert cmd==f'python3 {R}/{old}',cmd
 assert not (p/f'task/{pid}/children').read_text().strip()
 w=(p/'wchan').read_text();assert 'nanosleep' in w,w
 proof['coordinators'].append({'old_pid':pid,'old_cmdline':cmd,'old_wchan':w,'old_children':[],'new_source':new,'source_manifest_sha256':hashlib.sha256((R/f'source-host-r{rev}.json').read_bytes()).hexdigest()})
for row,(pid,old,new,rev) in zip(proof['coordinators'],items):
 os.kill(pid,signal.SIGTERM);p=Path(f'/proc/{pid}')
 for _ in range(100):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.05)
 else:raise RuntimeError('Old coordinator still alive')
 row['old_terminal']=True
 with (R/f'queued-recovery-host-r{rev}.log').open('x') as f:
  proc=subprocess.Popen(['python3',str(R/new)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(1);assert proc.poll() is None
  q=Path(f'/proc/{proc.pid}');assert not (q/f'task/{proc.pid}/children').read_text().strip()
  assert 'nanosleep' in (q/'wchan').read_text()
  row.update(new_pid=proc.pid,new_waiting_verified=True)
 out.write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
