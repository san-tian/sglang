"""Full recovery on a new immutable image; never write original R31 success barriers."""
from host_common_r118 import *
manifest=json.loads((R/'source-host-r118.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
probe=json.loads((R/'dcp-candidate-probe-finished-r117.json').read_text());assert probe['passed'] and probe['runtime_image_id']==manifest['image']
assert not Path('/proc/954883').exists()
waitleases(['dcp-candidate-probe-finished-r117.json'],list(range(8)),timeout=172800)
assert not (R/'eight-gpu-recovery-complete-r118.json').exists()
assert shutil.disk_usage('/').free>4*1024**3
(R/'eight-gpu-recovery-started-r118.json').write_text(json.dumps({'image':manifest['image'],'scope':'New complete DCP2 and attentionDP8 cohorts. Existing original TP8 baseline retained, original failed DCP mixed retained. Whole goal remains incomplete.'}))

def mixed(name,top,port,early=False):
 suffix='early-mixed' if early else 'mixed'
 adapters=[{'name':n,'path':ADAPTERS+path} for n,path in [('real','real_r16'),('zero','zero'),('gdn','gdn_fused')]]
 config=R/f'{top}-{suffix}-adapters-r118.json';assert not config.exists();config.write_text(json.dumps(adapters))
 out=f'{top}-{suffix}-r118';captures=f'{top}-{suffix}-captures-r118'
 run(['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_mixed.py','--url',f'http://127.0.0.1:{port}','--adapters',W+'/'+config.name,'--corpus','/work/qwen38-mol-speed-20261019/corpus-mixed.json','--capture-dir',W+'/'+captures,'--output',W+'/'+out,'--cases','4','--decode-tokens','16','--rounds','3'],out+'.log')
 s=json.loads((R/out/'summary.json').read_text())
 assert s['mixed_requests']==48 and all(v['passes_logprob_gate'] for v in s['scores'].values())
 return s

# Fail fast on the exact original serving failure, before the long full protocol.
name='qwen38-dcp8x2-d-early-r118';port=30702
launch(name,list(range(8)),8,port,BASE,extra=['--dcp-size','2','--dcp-comm-backend','a2a'])
try:
 health(port,name);placement(name);mixed(name,'dcp8x2',port,early=True)
finally:stop([name])

for top,dpa,extra in [('dcp8x2',False,['--dcp-size','2','--dcp-comm-backend','a2a']),('dpa8',True,[])]:
 port=30702 if top=='dcp8x2' else 30704;mamba=32 if dpa else 256
 for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
  name=f'qwen38-{top}-m-{adapter}-r118'
  launch(name,list(range(8)),8,port,MERGES+adapter,dpa=dpa,extra=extra,mamba=mamba)
  try:
   health(port,name);placement(name)
   compare(name,'reference',label,port,top,adapter,**({} if label=='full' else {'limit':4,'nt':16,'rounds':1,'batch':4}))
   if adapter=='real_r16':perf(name,port,top,'m')
  finally:stop([name])
 name=f'qwen38-{top}-d-r118';launch(name,list(range(8)),8,port,BASE,dpa=dpa,extra=extra,mamba=mamba)
 try:
  health(port,name);placement(name)
  for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
   result=compare(name,'dynamic',label,port,top,adapter)
   assert result['case_pairs']==(240 if label=='full' else 8)
   assert all(v['passes_logprob_gate'] for v in result['scores'].values())
  mixed(name,top,port);perf(name,port,top,'d')
 finally:stop([name])
 comparison=compare_perf(top);assert comparison['outputs_equal'],comparison
 name=f'qwen38-{top}-base-r118';launch(name,list(range(8)),8,port,BASE+'/',dpa=dpa,extra=extra,mamba=mamba)
 try:health(port,name);placement(name);perf(name,port,top,'base')
 finally:stop([name])
 for side in ['m','d','base']:
  directory=R/f'perf-{top}-{side}-r118'
  run(['python3',str(R/'analyze_profiles_r30.py'),str(directory/'profile'),'--output',str(directory/'kernel-summary.json')])
 name=f'qwen38-{top}-capacity-r118';physical_rows=8191 if dpa else 32767
 capextra=extra+['--page-size','1','--context-length','16384','--max-total-tokens',str(physical_rows),'--enable-metrics','--decode-log-interval','1']
 launch(name,list(range(8)),8,port,BASE,dpa=dpa,extra=capextra,mamba=mamba)
 try:
  health(port,name);placement(name)
  run(['docker','exec',name,'python3',W+'/capacity_residency_r61.py','--url',f'http://127.0.0.1:{port}','--adapter-path',ADAPTERS+'real_r16','--output',W+f'/capacity-{top}-r118'],f'capacity-{top}-r118.log')
 finally:stop([name])
 (R/f'{top}-precision-perf-capacity-complete-r118.json').write_text(json.dumps({'precision_pairs':240,'gdn_pairs':8,'mixed_requests':48,'performance':comparison,'raw_protocol_complete':True,'independent_raw_audit_pending':True,'full_goal_complete':False}))
(R/'eight-gpu-recovery-complete-r118.json').write_text(json.dumps({'topologies':['dcp8x2','dpa8'],'existing_tp8_baseline':'tp8-original-audits-finished-r102.json','image':manifest['image'],'independent_raw_audits_pending':True,'full_goal_complete':False}))
