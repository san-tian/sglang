"""Native 4P4D independent-merge accuracy and natural-SSE performance on all 8 GPUs."""
from host_common_r86 import *
manifest=json.loads((R/'source-host-r86.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
waitleases(['eight-gpu-suite-complete-r31.json','pd1-prototype-complete-r30.json'],list(range(8)),timeout=172800)
# UCX can use CUDA IPC over the single host's NVLink domain. Record actual
# transport logs and kernels: an environment selection is not transfer proof.
env={'SGLANG_DISAGGREGATION_NIXL_BACKEND':'UCX','UCX_TLS':'cuda_copy,cuda_ipc,posix,sysv,tcp','UCX_LOG_LEVEL':'info'}
portp,portd,bootstrap=30800,30801,30810
def pair(label,model):
 names=[]
 for mode,port,gpus,nccl in [('prefill',portp,list(range(4)),30820),('decode',portd,list(range(4,8)),30821)]:
  name=f'qwen38-pd4-{label}-{mode}-r33'
  extra=['--disaggregation-mode',mode,'--disaggregation-bootstrap-port',str(bootstrap),'--nccl-port',str(nccl),'--disaggregation-transfer-backend','nixl','--mamba-radix-cache-strategy','extra_buffer','--enable-metrics','--disaggregation-enable-kv-checksum','--watchdog-timeout','900']
  launch(name,gpus,4,port,model,extra=extra,env=env);names.append(name)
 return names
def performance(names,side):
 out=f'perf-pd4-{side}-r33'
 cmd=['docker','exec','-e','PYTHONPATH='+CLIENT,names[1],'python3',W+'/perf_pd_r81.py','--prefill-url',f'http://127.0.0.1:{portp}','--url',f'http://127.0.0.1:{portd}','--bootstrap-port',str(bootstrap),'--output',W+'/'+out,'--profile']
 if side=='d':cmd+=['--adapter-path',ADAPTERS+'real_r16']
 smoke=cmd[:];smoke.remove('--profile');smoke[smoke.index('--output')+1]=W+'/'+out+'-smoke';smoke+=['--smoke']
 run(smoke,out+'-smoke.log')
 run(cmd,out+'.log')
 run(['python3',str(R/'analyze_profiles_r30.py'),str(R/out/'profile'),'--output',str(R/out/'kernel-summary.json')])
for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
 names=pair('m-'+adapter,MERGES+adapter)
 try:
  for name,port in zip(names,[portp,portd]):health(port,name)
  compare(names[1],'reference',label,portd,'pd4-r33',adapter,**({'prefill':portp,'bootstrap':bootstrap} if label=='full' else {'limit':4,'nt':16,'rounds':1,'batch':4,'prefill':portp,'bootstrap':bootstrap}))
  if adapter=='real_r16':performance(names,'m')
 finally:stop(names)
names=pair('d',BASE)
try:
 for name,port in zip(names,[portp,portd]):health(port,name)
 for adapter,label in [('real_r16','full'),('gdn_fused','gdn')]:
  summary=compare(names[1],'dynamic',label,portd,'pd4-r33',adapter,prefill=portp,bootstrap=bootstrap)
  assert summary['case_pairs']==(240 if label=='full' else 8) and all(v['passes_logprob_gate'] for v in summary['scores'].values())
 performance(names,'d')
finally:stop(names)
left=[json.loads(x) for x in (R/'perf-pd4-d-r33/requests.jsonl').read_text().splitlines()]
right={r['label']:r for r in [json.loads(x) for x in (R/'perf-pd4-m-r33/requests.jsonl').read_text().splitlines()]}
assert len(left)==len(right)==636
wrong=[r['label'] for r in left if r['output_ids']!=right[r['label']]['output_ids']]
(R/'pd4-natural-comparison-r39.json').write_text(json.dumps({'requests_per_side':636,'mismatched_labels':wrong,'outputs_equal':not wrong},indent=2))
assert not wrong,wrong
names=pair('base',BASE+'/')
try:
 for name,port in zip(names,[portp,portd]):health(port,name)
 performance(names,'base')
finally:stop(names)
(R/'pd4-precision-perf-complete-r33.json').write_text(json.dumps({'prefill_gpus':[0,1,2,3],'decode_gpus':[4,5,6,7],'real_pairs':240,'gdn_pairs':8,'natural_requests_per_side':636,'base_performance_requests':636,'natural_outputs_equal':not wrong,'mismatched_labels':wrong,'transport':'NIXL/UCX with KV+SSM checksum enabled; inspect actual logs/traces before claiming transfer cost or peak bandwidth'}))
