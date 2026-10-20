"""Audit complete new topology evidence; a live coordinator is checked while waiting."""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
source=json.loads((R/'source-attention-dp8-audits-r130.json').read_text())
record=json.loads((R/'attention-dp8-process-r130.json').read_text());pid=record['pid']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 for n,h in source['source_files'].items():assert sha(R/n)==h,n
verify();deadline=time.monotonic()+172800
result={'passed':False,'full_goal_complete':False,'topologies':{},'runtime_image':source['runtime_image'],'scope':'Original-image attentionDP8 full raw audits, not cached optimization,4P4D or overall goal completion.'}
def wait(path):
 while not path.exists():
  proc=Path(f'/proc/{pid}')
  assert proc.exists() and (proc/'stat').read_text().split()[2]!='Z',('Coordinator terminal without required evidence',path.name,pid)
  assert b'run_attention_dp8_r130.py' in (proc/'cmdline').read_bytes()
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; GPU restart not authorized by this timeout')
  time.sleep(5)
def run(script,args,output,log):
 assert not (R/output).exists();verify()
 with (R/log).open('x') as f:subprocess.run(['python3',str(R/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 x=json.loads((R/output).read_text());assert x['passed'];return x
try:
 for top in ['dpa8']:
  wait(R/f'{top}-precision-perf-capacity-complete-r130.json')
  pn=f'{top}-precision-raw-audit-r130.json'
  p=run('audit_attention_dp8_precision_r130.py',['--root',str(R),'--topology',top,'--output',str(R/pn)],pn,f'{top}-precision-raw-audit-r130.log')
  assert (p['case_pairs'],p['score_count'])==(248,114692)
  nn=f'{top}-native-kernel-raw-audit-r130.json'
  n=run('audit_attention_dp8_native_r130.py',[top],nn,f'{top}-native-kernel-raw-audit-r130.log')
  assert (n['raw_native_payloads_audited'],n['latency_statistics_recomputed'])==(1908,576)
  mn=f'{top}-mixed-raw-audit-r130.json'
  m=run('audit_attention_dp8_mixed_r130.py',['--root',str(R),'--topology',top,'--output',str(R/mn)],mn,f'{top}-mixed-raw-audit-r130.log')
  assert m['mixed_requests']==48 and sum(s['count'] for s in m['scores'].values())==768
  cn=f'capacity-{top}-independent-audit-r130.json'
  c=run('audit_attention_dp8_capacity_r130.py',[top,'--root',str(R)],cn,f'{top}-capacity-raw-audit-r130.log')
  assert c['native_requests_audited']==13 and c['full_attention_KV_bytes_per_rank']==536870912
  expected=[f'qwen38-{top}-{role}-r130' for role in ['m-real_r16','m-gdn_fused','d','base','capacity']]
  observations=[json.loads(s) for s in (R/'actual-placement-observations-r130.jsonl').read_text().splitlines()]
  placement={}
  for name in expected:
   matches=[x for x in observations if x['name']==name];assert len(matches)==1
   x=matches[0];assert x['placement_matched'] and x['image']==source['runtime_image'] and x['observed_compute_gpus']==x['expected_compute_gpus']==list(range(8))
   assert len({p['pid'] for p in x['scheduler_gpu_processes']})==8
   placement[name]={'container_id':x['container_id'],'gpus':x['observed_compute_gpus']}
  detail={'passed':True,'full_goal_complete':False,'reports_sha256':{f:sha(R/f) for f in [pn,nn,mn,cn]},'actual_placement':placement}
  with (R/f'{top}-raw-audits-finished-r130.json').open('x') as f:json.dump(detail,f,indent=2)
  result['topologies'][top]=detail;print('audited',top,flush=True)
 result['passed']=True
except BaseException:result['error']=traceback.format_exc();raise
finally:
 result['at_unix']=time.time()
 with (R/'attention-dp8-raw-audits-finished-r130.json').open('x') as f:json.dump(result,f,indent=2)
