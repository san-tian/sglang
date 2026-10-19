"""Replace only a verified childless sleeping CPU coordinator before GPU lease."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
out=R/'million-coordinator-migration-r106.json';assert not out.exists()
manifest=R/'source-milli-cache-r106.json'
for n,h in json.loads(manifest.read_text())['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
pid=996223;p=Path(f'/proc/{pid}')
def guards():
 assert (p/'cmdline').read_bytes().split(b'\0')[:-1]==[b'python3',b'-u',str(R/'run_million_cache_r97.py').encode()]
 assert not (p/f'task/{pid}/children').read_text().strip()
 assert 'nanosleep' in (p/'wchan').read_text()
 for name in ['isolated-speed-finished-r69.json','million-cache-finished-r97.json','million-cache-gpu-observations-r97.jsonl','million-cache-comparison-r97.json']:
  assert not (R/name).exists(),name
 assert not list(R.glob('perf-million-*-r97'))
 assert not list(R.glob('launch-qwen38-million-*-r97.json'))
 assert not list((R/'caches').glob('qwen38-million-*-r97'))
 names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
 assert not any(n.startswith('qwen38-million-') for n in names)
 return names
names=guards()
# Capture live TP8's identity before/after; this script never signals it or Docker.
container='qwen38-tp8-m-real_r16-r85'
before=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
assert before['State']['Running']
proof={'unix_s':time.time(),'old_pid':pid,'old_cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),
       'old_children':[],'old_wchan':(p/'wchan').read_text(),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
       'scope':'Only verified waiting childless CPU coordinator migrated; no million GPU arms started, no GPU worker signaled.',
       'runtime_workers_interrupted':False,'tp8_container_id':before['Id'],'tp8_started_at':before['State']['StartedAt']}
guards();os.kill(pid,signal.SIGTERM)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.05)
else:raise RuntimeError('Old coordinator remains live')
proof['old_terminal']=True
with (R/'million-cache-host-r106.log').open('x') as log:
 proc=subprocess.Popen(['python3','-u',str(R/'run_million_cache_r106.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(1);assert proc.poll() is None
 q=Path(f'/proc/{proc.pid}');assert not (q/f'task/{proc.pid}/children').read_text().strip()
 assert 'nanosleep' in (q/'wchan').read_text()
 proof.update(new_pid=proc.pid,new_cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),new_wchan=(q/'wchan').read_text(),new_waiting_verified=True)
after=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
assert after['Id']==before['Id'] and after['State']['StartedAt']==before['State']['StartedAt'] and after['State']['Running']
proof['tp8_continued_verified']=True
with out.open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof,indent=2))
