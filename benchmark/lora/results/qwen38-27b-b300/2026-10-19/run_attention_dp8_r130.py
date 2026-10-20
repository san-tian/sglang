"""Original-image attentionDP8 continuation. DCP numerical failure remains rejected."""
from host_common_r130 import *
manifest=json.loads((R/'source-host-r130.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert not Path('/proc/1264524').exists()
assert not (R/'eight-gpu-recovery-complete-r118.json').exists()
assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False
waitleases(['real-qkv-oracle-state-r129.json'],list(range(8)),timeout=172800)
oracle=json.loads((R/'real-qkv-oracle-state-r129.json').read_text())
assert oracle['returncode']==0 and not oracle['state']['Running']
assert json.loads((R/'real-qkv-r129/oracle-finished.json').read_text())['diagnosis_completed']
assert shutil.disk_usage('/').free>4*1024**3
assert not (R/'attention-dp8-resource-released-r130.json').exists()
try:
 def mixed(name,top,port,early=False):
  suffix='early-mixed' if early else 'mixed'
  adapters=[{'name':n,'path':ADAPTERS+path} for n,path in [('real','real_r16'),('zero','zero'),('gdn','gdn_fused')]]
  config=R/f'{top}-{suffix}-adapters-r130.json';assert not config.exists();config.write_text(json.dumps(adapters))
  out=f'{top}-{suffix}-r130';captures=f'{top}-{suffix}-captures-r130'
  run(['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_mixed.py','--url',f'http://127.0.0.1:{port}','--adapters',W+'/'+config.name,'--corpus','/work/qwen38-mol-speed-20261019/corpus-mixed.json','--capture-dir',W+'/'+captures,'--output',W+'/'+out,'--cases','4','--decode-tokens','16','--rounds','3'],out+'.log')
  s=json.loads((R/out/'summary.json').read_text())
  assert s['mixed_requests']==48 and all(v['passes_logprob_gate'] for v in s['scores'].values())
  return s
 
 (R/'attention-dp8-cohort-complete-r130.json').write_text(json.dumps({'topologies':['dpa8'],'image':manifest['image'],'raw_protocol_complete':True,'independent_raw_audits_pending':True,'full_goal_complete':False}))
finally:
 # This releases only an experiment resource lease, including on failure.
 # It never asserts model correctness or substitutes for R31/R118 success.
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if not apps.strip():
  state={'at_unix':time.time(),'resource_released':True,'observed_gpu_processes':apps,
         'cohort_completed':(R/'attention-dp8-cohort-complete-r130.json').exists(),
         'model_accuracy_accepted':False,'full_goal_complete':False,
         'scope':'GPU lease release only; no correctness/speed/capacity acceptance.'}
  with (R/'attention-dp8-resource-released-r130.json').open('x') as f:json.dump(state,f,indent=2)
