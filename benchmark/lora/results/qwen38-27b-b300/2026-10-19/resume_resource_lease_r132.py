"""Run an unchanged queued experiment with one explicitly versioned lease barrier.

No original success file is created. Runtime, clients and measurements are the
original sealed sources; the migration sidecar records the new effective wait.
"""
import argparse,hashlib,importlib,json,runpy,time
from pathlib import Path
R=Path(__file__).resolve().parent
JOBS={
 'pd':('run_pd4_suite_r86.py','host_common_r86','source-host-r86.json'),
 'cached':('run_cached_topologies_r90.py','host_common_r90','source-host-r90.json'),
 'isolated':('run_isolated_speed_r91.py','host_common_r90','source-host-r91.json'),
 'million':('run_million_cache_r106.py','host_common_r90','source-milli-cache-r106.json'),
}
OLD='eight-gpu-suite-complete-r31.json'
NEW='attention-dp8-resource-released-r131.json'
def mapped(markers):
 assert OLD in markers and NEW not in markers
 return [NEW if x==OLD else x for x in markers]
def main(kind):
 source=json.loads((R/'source-resource-queue-r132.json').read_text())
 for n,h in source['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 driver,modname,parent=JOBS[kind]
 original=json.loads((R/parent).read_text())
 for n,h in original['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 module=importlib.import_module(modname)
 original_wait=module.waitleases
 def wait(markers,gpus,timeout=18000):
  effective=mapped(markers)
  with (R/f'{kind}-effective-wait-barriers-r132.json').open('x') as f:
   json.dump({'at_unix':time.time(),'original_driver':driver,'original_manifest':parent,'original_markers':markers,'effective_markers':effective,'gpus':gpus,'timeout':timeout,'original_measurement_source_unmodified':True,'full_goal_complete':False},f,indent=2)
  original_wait(effective,gpus,timeout)
  resource=json.loads((R/NEW).read_text())
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
 module.waitleases=wait
 runpy.run_path(str(R/driver),run_name='__main__')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=JOBS,required=True);a=p.parse_args();main(a.kind)
