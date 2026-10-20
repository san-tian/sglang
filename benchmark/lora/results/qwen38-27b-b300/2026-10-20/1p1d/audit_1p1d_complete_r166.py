"""Run raw audit and distinguish configured request limits from scheduler limits."""
import hashlib,json,re,time
from pathlib import Path
from audit_1p1d_r162 import inspect
R=Path(__file__).resolve().parent
def main():
 finished=json.loads((R/'1p1d-control-finished-r161.json').read_text());assert finished['passed'] and finished['primary_dynamic_restored']
 source=json.loads((R/'source-1p1d-r161.json').read_text())
 for n,h in source['source_files_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 report=inspect(R)
 report['audited_at_unix_s']=time.time()
 for role,allocation in report['allocated_capacity_per_role'].items():
  text=(R/f'qwen38-1p1d-{role}-r160-before-merge-r161.log').read_text()
  matches=re.findall(r'max_total_num_tokens=\d+.*max_running_requests=(\d+), context_len=',text)
  assert len(matches)==1,matches
  allocation['max_running_requests_configured']=allocation.pop('max_running_requests_resolved')
  allocation['max_running_requests_scheduler_resolved']=int(matches[0])
  state=re.search(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+), conv_state size: ([0-9.]+)GB, ssm_state size: ([0-9.]+)GB',text);assert state
  allocation['Mamba_conv_GiB_rounded']=float(state[2]);allocation['Mamba_SSM_GiB_rounded']=float(state[3])
  allocation['merged_projection_cache_MiB']=36864
 report['shared_request_KV_pool_tokens_bound']=min(v['actual_allocated_KV_tokens'] for v in report['allocated_capacity_per_role'].values())
 report['teacher_forced_precision']=finished['scores']
 assert all(v['passes_logprob_gate'] for v in finished['scores']['scores'].values())
 report['source_files_sha256']=source['source_files_sha256']
 report['full_goal_complete']=False
 with (R/'1p1d-main-audit-r166.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:v for k,v in report.items() if k in ['passed','speed','allocated_capacity_per_role','capacity_residency','slots','teacher_forced_precision','dynamic_vs_merge']}))
if __name__=='__main__':main()
