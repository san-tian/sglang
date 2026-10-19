"""Read-only actual GPU/cgroup placement evidence for queued all8 and full4P4D.

Do not infer placement from visibility or base-gpu-id. Match docker cgroup host
PIDs to live nvidia-smi GPU UUIDs. This proves observed compute placement only,
not numerical accuracy, transport path, KV capacity or performance.
"""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
IMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
def cmd(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
def expected(name):
 if name.startswith('qwen38-pd4-'):
  return set(range(4)) if '-prefill-' in name else set(range(4,8)) if '-decode-' in name else None
 if name.startswith(('qwen38-tp8-','qwen38-dcp8x2-','qwen38-dpa8-')):return set(range(8))
 return None

def inspect_placement(name,gpus,apps,info,top):
 processes=[]
 for line in top.strip().splitlines()[1:]:
  vals=line.strip().split(None,2)
  if len(vals)==3 and vals[0].isdigit():processes.append({'pid':int(vals[0]),'ppid':int(vals[1]),'args':vals[2]})
 sched={p['pid'] for p in processes if 'scheduler' in p['args'].lower()}
 actual=[{**p,'gpu_index':gpus[p['gpu_uuid']]} for p in apps if p['pid'] in sched]
 wanted=expected(name);observed={p['gpu_index'] for p in actual}
 return {'unix_s':time.time(),'name':name,'container_id':info['Id'],'image':info['Image'],
         'container_running':info['State']['Running'],'command':info['Config']['Cmd'],
         'docker_cgroup_host_processes':processes,'scheduler_gpu_processes':actual,
         'all_container_gpu_processes':[p for p in apps if p['pid'] in {q['pid'] for q in processes}],
         'expected_compute_gpus':sorted(wanted),'observed_compute_gpus':sorted(observed),
         'placement_matched':bool(sched) and wanted==observed and info['Image']==IMAGE,
         'scope':'Actual scheduler host PID to live GPU UUID match; no model/transport/performance acceptance.'}

def main():
 path=R/'actual-placement-observations-r92.jsonl';assert not path.exists()
 completed={};errors=0;end=time.monotonic()+172800
 source={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'runtime_image':IMAGE,'read_only':True,'scopes':['TP8','DCP2','attentionDP8','full4P4D'],'started_unix_s':time.time()}
 (R/'actual-placement-source-r92.json').write_text(json.dumps(source,indent=2))
 with path.open('x') as stream,(R/'actual-placement-errors-r92.jsonl').open('x') as err:
  while time.monotonic()<end:
   try:
    gs=cmd(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'])
    gpus={v[1].strip():int(v[0]) for v in [s.split(',') for s in gs.splitlines()]}
    raw=cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
    apps=[{'pid':int(v[0]),'gpu_uuid':v[1].strip(),'memory_mib':v[2].strip()} for v in [s.split(',') for s in raw.splitlines()] if len(v)==3]
    names=cmd(['docker','ps','--format','{{.Names}}']).splitlines()
    for name in names:
     if expected(name) is None or name in completed:continue
     info=json.loads(cmd(['docker','inspect',name]))[0]
     if not info['State']['Running']:continue
     top=cmd(['docker','top',name,'-eo','pid,ppid,args'])
     row=inspect_placement(name,gpus,apps,info,top)
     if row['placement_matched']:
      # Keep one full successful snapshot per immutable container, not repeated telemetry.
      stream.write(json.dumps(row)+'\n');stream.flush()
      completed[name]={'container_id':row['container_id'],'expected':row['expected_compute_gpus'],'observed':row['observed_compute_gpus'],'unix_s':row['unix_s']}
      print('matched',name,row['observed_compute_gpus'],flush=True)
    if (R/'eight-gpu-suite-complete-r31.json').exists() and (R/'pd4-precision-perf-complete-r33.json').exists():break
   except Exception:
    errors+=1;err.write(json.dumps({'unix_s':time.time(),'error':traceback.format_exc()})+'\n');err.flush()
   time.sleep(5)
 (R/'actual-placement-finished-r92.json').write_text(json.dumps({'observations':completed,'poll_errors':errors,'timeout':time.monotonic()>=end,'scope':source['scopes'],'all8_and_PD_completion_markers_present':all((R/p).exists() for p in ['eight-gpu-suite-complete-r31.json','pd4-precision-perf-complete-r33.json']),'full_goal_complete':False},indent=2))
if __name__=='__main__':main()
