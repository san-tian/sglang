"""New namespaces and provenance for the full DCP2/attentionDP8 recovery."""
import ast,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
IMAGE='sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878'
BASE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
old=(R/'host_common_r85.py').read_text()
new=old.replace(BASE,IMAGE).replace('source-host-r85.json','source-host-r118.json')
new=new.replace('-r30','-r118').replace("CLIENT=W+'/client-r118'","CLIENT=W+'/client-r30'")
# File names analyze_profiles_r30/perf_endpoint_r85 and the launch template stay frozen.
assert 'launch-qwen38-tp2-d-r28.json' in new and "CLIENT=W+'/client-r30'" in new
# A ready service must provide actual scheduler PID -> GPU placement before tests.
pos=new.index('def launch(')
new=new[:pos]+'''def placement(name):
 import watch_actual_placement_r92 as observer
 observer.IMAGE='''+repr(IMAGE)+'''
 mapping={s.split(',')[1].strip():int(s.split(',')[0]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()}
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
 apps=[{'pid':int(v[0]),'gpu_uuid':v[1].strip(),'memory_mib':v[2].strip()} for v in [s.split(',') for s in raw.splitlines()] if len(v)==3]
 info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
 top=subprocess.check_output(['docker','top',name,'-eo','pid,ppid,args'],text=True)
 row=observer.inspect_placement(name,mapping,apps,info,top)
 assert row['placement_matched'] and len({p['pid'] for p in row['scheduler_gpu_processes']})==8,row
 with (R/'actual-placement-observations-r118.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
 return row

'''+new[pos:]
(R/'host_common_r118.py').write_text(new)
driver='''"""Full recovery on a new immutable image; never write original R31 success barriers."""
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
'''
(R/'run_eight_gpu_recovery_r118.py').write_text(driver)
files=list(json.loads((R/'source-host-r85.json').read_text())['source_files'])
files=[n for n in files if n not in ['host_common_r85.py','run_eight_gpu_suite_r85.py']]
files += ['host_common_r118.py','run_eight_gpu_recovery_r118.py','watch_actual_placement_r92.py','source-runtime-candidate-r116.json','probe_dcp_chunk_invariance_r117.py']
for n in files:
 if n.endswith('.py'):ast.parse((R/n).read_text())
manifest={'image':IMAGE,'runtime_image':IMAGE,'base_image':BASE,'base_runtime_commit':'a589b336780051bbf26755cf1ea0aba56fbd029c','runtime_source_override_manifest':'source-runtime-candidate-r116.json','source_files':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files},'scope':'Complete new DCP2 and attentionDP8 protocol after preserved R85 mixed failure;248 D/M precision,48 mixed,1908 natural SSE,24 traces,13 equal-byte residency requests per topology; early exact-failure mixed screen. No original success barriers written. Full4P4D and cache/isolated/million followup remain required.','source_namespace':'r118'}
(R/'source-host-r118.json').write_text(json.dumps(manifest,indent=2))
print('Prepared full 2-topology protocol, original R30 clients, immutable source manifest. Not GPU acceptance.')
