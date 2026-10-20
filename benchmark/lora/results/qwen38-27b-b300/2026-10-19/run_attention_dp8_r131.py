"""Original-image attentionDP8 continuation. DCP numerical failure remains rejected."""
from host_common_r131 import *
manifest=json.loads((R/'source-host-r131.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert not Path('/proc/1264524').exists()
assert not (R/'eight-gpu-recovery-complete-r118.json').exists()
assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False
waitleases(['real-qkv-oracle-state-r129.json'],list(range(8)),timeout=172800)
oracle=json.loads((R/'real-qkv-oracle-state-r129.json').read_text())
assert oracle['returncode']==0 and not oracle['state']['Running']
assert json.loads((R/'real-qkv-r129/oracle-finished.json').read_text())['diagnosis_completed']
assert shutil.disk_usage('/').free>4*1024**3
assert not (R/'attention-dp8-resource-released-r131.json').exists()
try:
 def mixed(name,top,port,early=False):
  suffix='early-mixed' if early else 'mixed'
  adapters=[{'name':n,'path':ADAPTERS+path} for n,path in [('real','real_r16'),('zero','zero'),('gdn','gdn_fused')]]
  config=R/f'{top}-{suffix}-adapters-r131.json';assert not config.exists();config.write_text(json.dumps(adapters))
  out=f'{top}-{suffix}-r131';captures=f'{top}-{suffix}-captures-r131'
  run(['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_mixed.py','--url',f'http://127.0.0.1:{port}','--adapters',W+'/'+config.name,'--corpus','/work/qwen38-mol-speed-20261019/corpus-mixed.json','--capture-dir',W+'/'+captures,'--output',W+'/'+out,'--cases','4','--decode-tokens','16','--rounds','3'],out+'.log')
  s=json.loads((R/out/'summary.json').read_text())
  assert s['mixed_requests']==48 and all(v['passes_logprob_gate'] for v in s['scores'].values())
  return s
 
 for top,dpa,extra in [('dpa8',True,[])]:
  port=30702 if top=='dcp8x2' else 30704;mamba=32 if dpa else 256
  for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
   name=f'qwen38-{top}-m-{adapter}-r131'
   launch(name,list(range(8)),8,port,MERGES+adapter,dpa=dpa,extra=extra,mamba=mamba)
   try:
    health(port,name);placement(name)
    compare(name,'reference',label,port,top,adapter,**({} if label=='full' else {'limit':4,'nt':16,'rounds':1,'batch':4}))
    if adapter=='real_r16':perf(name,port,top,'m')
   finally:stop([name])
  name=f'qwen38-{top}-d-r131';launch(name,list(range(8)),8,port,BASE,dpa=dpa,extra=extra,mamba=mamba)
  try:
   health(port,name);placement(name)
   for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
    result=compare(name,'dynamic',label,port,top,adapter)
    assert result['case_pairs']==(240 if label=='full' else 8)
    assert all(v['passes_logprob_gate'] for v in result['scores'].values())
   mixed(name,top,port);perf(name,port,top,'d')
  finally:stop([name])
  comparison=compare_perf(top);assert comparison['outputs_equal'],comparison
  name=f'qwen38-{top}-base-r131';launch(name,list(range(8)),8,port,BASE+'/',dpa=dpa,extra=extra,mamba=mamba)
  try:health(port,name);placement(name);perf(name,port,top,'base')
  finally:stop([name])
  for side in ['m','d','base']:
   directory=R/f'perf-{top}-{side}-r131'
   run(['python3',str(R/'analyze_profiles_r30.py'),str(directory/'profile'),'--output',str(directory/'kernel-summary.json')])
  name=f'qwen38-{top}-capacity-r131';physical_rows=8191 if dpa else 32767
  capextra=extra+['--page-size','1','--context-length','16384','--max-total-tokens',str(physical_rows),'--enable-metrics','--decode-log-interval','1']
  launch(name,list(range(8)),8,port,BASE,dpa=dpa,extra=capextra,mamba=mamba)
  try:
   health(port,name);placement(name)
   run(['docker','exec',name,'python3',W+'/capacity_residency_r61.py','--url',f'http://127.0.0.1:{port}','--adapter-path',ADAPTERS+'real_r16','--output',W+f'/capacity-{top}-r131'],f'capacity-{top}-r131.log')
  finally:stop([name])
  (R/f'{top}-precision-perf-capacity-complete-r131.json').write_text(json.dumps({'precision_pairs':240,'gdn_pairs':8,'mixed_requests':48,'performance':comparison,'raw_protocol_complete':True,'independent_raw_audit_pending':True,'full_goal_complete':False}))
 item=json.loads((R/'dpa8-precision-perf-capacity-complete-r131.json').read_text())
 assert item['raw_protocol_complete'] and (item['precision_pairs'],item['gdn_pairs'],item['mixed_requests'])==(240,8,48)
 assert item['performance']['requests_per_side']==636 and item['performance']['outputs_equal']
 for side in ['m','d','base']:
  complete=json.loads((R/f'perf-dpa8-{side}-r131/complete.json').read_text())
  assert complete['requests']==636 and complete['profile']
 for label in ['full','gdn']:
  for stage in ['reference','dynamic']:
   directory=R/f'dpa8-{label}-{stage}-r131'
   seal=json.loads((directory/'seal.json').read_text())
   assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==h for name,h in seal['files'].items())
 rows=[json.loads(line) for line in (R/'actual-placement-observations-r131.jsonl').read_text().splitlines()]
 assert {x['name'] for x in rows}=={f'qwen38-dpa8-{role}-r131' for role in ['m-real_r16','m-gdn_fused','d','base','capacity']}
 assert all(x['placement_matched'] for x in rows)
 (R/'attention-dp8-cohort-complete-r131.json').write_text(json.dumps({'topologies':['dpa8'],'image':manifest['image'],'raw_protocol_complete':True,'independent_raw_audits_pending':True,'full_goal_complete':False}))
finally:
 # This releases only an experiment resource lease, including on failure.
 # It never asserts model correctness or substitutes for R31/R118 success.
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if not apps.strip():
  state={'at_unix':time.time(),'resource_released':True,'observed_gpu_processes':apps,
         'cohort_completed':(R/'attention-dp8-cohort-complete-r131.json').exists(),
         'model_accuracy_accepted':False,'full_goal_complete':False,
         'scope':'GPU lease release only; no correctness/speed/capacity acceptance.'}
  with (R/'attention-dp8-resource-released-r131.json').open('x') as f:json.dump(state,f,indent=2)
