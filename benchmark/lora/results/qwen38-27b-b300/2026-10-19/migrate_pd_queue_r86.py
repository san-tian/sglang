"""Replace only the childless waiting R78 PD coordinator to bound profiler stack/shape collection."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;oldpid=944959
out=R/'pd4-queue-migration-r86.json';assert not out.exists()
m=json.loads((R/'source-host-r86.json').read_text())
for n,h in m['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
transport=R/'nixl-transport-controls-audit-r78.json';audit=json.loads(transport.read_text())
assert audit['passed'] and audit['controls']['76']['direct_PtoP_observed']
assert all(not audit['controls'][str(i)]['direct_PtoP_observed'] for i in [74,75,77])
assert not (R/'eight-gpu-suite-complete-r31.json').exists()
assert not (R/'pd4-precision-perf-complete-r33.json').exists()
assert not list(R.glob('perf-pd4-*-r33'))
assert not subprocess.check_output(['docker','ps','-a','--filter','name=qwen38-pd4-','--format','{{.Names}}'],text=True).strip()
p=Path(f'/proc/{oldpid}')
cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode().strip()
assert cmd==f'python3 {R}/run_pd4_suite_r81.py',cmd
assert not (p/f'task/{oldpid}/children').read_text().strip()
wchan=(p/'wchan').read_text();assert 'nanosleep' in wchan,wchan
proof={'old_pid':oldpid,'old_cmdline':cmd,'old_wchan':wchan,'old_children':[],
       'old_source_manifest_sha256':hashlib.sha256((R/'source-host-r81.json').read_bytes()).hexdigest(),
       'new_source_manifest_sha256':hashlib.sha256((R/'source-host-r86.json').read_bytes()).hexdigest(),
       'actual_transport_audit_sha256':hashlib.sha256(transport.read_bytes()).hexdigest(),
       'changes':['Explicit profiler with_stack=False and record_shapes=False; CPU/GPU activities retained','TMPDIR and Python pycache redirected to per-container NVMe cache','Source provenance updated to R86'],
       'unchanged':['TP4 per role','KV+SSM checksum','extra_buffer strategy','NIXL/UCX and transport flags','M real240/GDN8/D matching pairs','D/M/base each636 native speed controls','Both role CUDA traces and metrics','All natural sampling and SSE timing code AST unchanged','P2/D10 profile steps unchanged','Stop both roles together','Existing all8-before-PD and subsequent cached/isolated barriers'],
       'runtime_workers_interrupted':False,'full_4P4D_accepted':False}
os.kill(oldpid,signal.SIGTERM)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.05)
else:raise RuntimeError('Old coordinator remains alive; no replacement started')
proof['old_coordinator_terminal']=True
assert not (R/'eight-gpu-suite-complete-r31.json').exists()
with (R/'pd4-suite-host-r86.log').open('x') as f:
 proc=subprocess.Popen(['python3',str(R/'run_pd4_suite_r86.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(1);assert proc.poll() is None
 p=Path(f'/proc/{proc.pid}');assert not (p/f'task/{proc.pid}/children').read_text().strip()
 assert 'nanosleep' in (p/'wchan').read_text()
 proof.update(new_pid=proc.pid,new_waiting_verified=True,unix_s=time.time())
out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
