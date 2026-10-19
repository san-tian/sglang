"""Resume full all8 protocol after terminal ENOSPC; preserve failed R63 artifacts."""
from host_common_r85 import *
manifest=json.loads((R/'source-host-r85.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
waitleases(['tp4-suite-complete-r30.json','cache-suite-complete-r34.json','cache-mixed-finished-r35.json','cache-extra-finished-r39.json','multigpu-base-complete-r41.json','fa4-suite-finished-r44.json','active-cache-recovery-finished-r84.json'],list(range(8)),timeout=172800)
for marker in ['cache-mixed-finished-r35.json','cache-extra-finished-r39.json']:
 assert json.loads((R/marker).read_text())['passed'],marker
for top in ['tp4','dpa4']:
 comparison=json.loads((R/f'perf-{top}-comparison-r30.json').read_text())
 assert comparison['outputs_equal'] and not comparison['mismatched_labels'],comparison
(R/'eight-gpu-suite-started-r85.json').write_text(json.dumps({'gpus':list(range(8)),'image':'qwen38-mol:r28'}))
assert shutil.disk_usage('/').free > 4*1024**3, 'Root filesystem needs recovery before GPU launch'
for top,dpa,extra in [('tp8',False,[]),('dcp8x2',False,['--dcp-size','2','--dcp-comm-backend','a2a']),('dpa8',True,[])]:
 port={'tp8':30700,'dcp8x2':30702,'dpa8':30704}[top];mamba=32 if dpa else 256
 for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
  name=f'qwen38-{top}-m-{adapter}-'+('r85' if top=='tp8' and adapter=='real_r16' else 'r31');launch(name,list(range(8)),8,port,MERGES+adapter,dpa=dpa,extra=extra,mamba=mamba);health(port,name)
  compare(name,'reference',label,port,top,adapter,**({} if label=='full' else {'limit':4,'nt':16,'rounds':1,'batch':4}))
  if adapter=='real_r16':perf(name,port,top,'m')
  stop([name])
 name=f'qwen38-{top}-d-r31';launch(name,list(range(8)),8,port,BASE,dpa=dpa,extra=extra,mamba=mamba);health(port,name)
 for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
  result=compare(name,'dynamic',label,port,top,adapter);assert result['case_pairs']==(240 if label=='full' else 8)
  assert all(v['passes_logprob_gate'] for v in result['scores'].values())
 adapters=[{'name':n,'path':ADAPTERS+path} for n,path in [('real','real_r16'),('zero','zero'),('gdn','gdn_fused')]]
 (R/f'{top}-mixed-adapters-r31.json').write_text(json.dumps(adapters))
 run(['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_mixed.py','--url',f'http://127.0.0.1:{port}','--adapters',W+f'/{top}-mixed-adapters-r31.json','--corpus','/work/qwen38-mol-speed-20261019/corpus-mixed.json','--capture-dir',W+f'/{top}-mixed-captures-r31','--output',W+f'/{top}-mixed-r31','--cases','4','--decode-tokens','16','--rounds','3'],f'{top}-mixed-r31.log')
 perf(name,port,top,'d');stop([name]);comparison=compare_perf(top);assert comparison['outputs_equal'],comparison
 basename=f'qwen38-{top}-base-r41';launch(basename,list(range(8)),8,port,BASE+'/',dpa=dpa,extra=extra,mamba=mamba);health(port,basename)
 try:perf(basename,port,top,'base')
 finally:stop([basename])
 for side in ['m','d','base']:
  directory=R/f'perf-{top}-{side}-r30'
  run(['python3',str(R/'analyze_profiles_r30.py'),str(directory/'profile'),'--output',str(directory/'kernel-summary.json')])
 # Equal BF16 tensor bytes/rank INCLUDING the one-row dummy page:
 # TP8/DCP2:32767+1 rows x one KV head; DPA8:8191+1 rows x four KV heads.
 capname=f'qwen38-{top}-capacity-r31';physical_rows=8191 if dpa else 32767
 capextra=extra+['--page-size','1','--context-length','16384','--max-total-tokens',str(physical_rows),'--enable-metrics','--decode-log-interval','1']
 launch(capname,list(range(8)),8,port,BASE,dpa=dpa,extra=capextra,mamba=mamba);health(port,capname)
 run(['docker','exec',capname,'python3',W+'/capacity_residency_r61.py','--url',f'http://127.0.0.1:{port}','--adapter-path',ADAPTERS+'real_r16','--output',W+f'/capacity-{top}-r63'],f'capacity-{top}-r31.log');stop([capname])
 (R/f'{top}-precision-perf-capacity-complete-r31.json').write_text(json.dumps({'precision_pairs':240,'gdn_pairs':8,'mixed_requests':48,'performance':comparison,'capacity':'capped-budget observations; see actual metrics and memory allocation'}))
(R/'eight-gpu-suite-complete-r31.json').write_text(json.dumps({'topologies':['tp8','dcp8x2','dpa8']}))
