"""Audit all stress SSE and decode metrics from the identical request window."""
import gzip,hashlib,json,time
from pathlib import Path
from audit_1p1d_r170 import metric
R=Path(__file__).resolve().parent;F=R/'capacity-1p1d-million-resident-r182'
while not (R/'1p1d-KV-stress-failed-r182.json').exists() and not (R/'1p1d-KV-stress-audit-r183.json').exists():time.sleep(3)
assert json.loads((F/'complete.json').read_text())['passed']
rows=[json.loads(s) for s in (F/'requests.jsonl').read_text().splitlines()];assert len(rows)==32
for row in rows:
 x=json.loads((F/(row['label']+'.json')).read_text());last=x['decode_events'][-1]['data']
 assert len(x['decode_request']['input_ids'])==32768
 assert row['output_ids']==last['output_ids'] and len(last['output_ids'])==2048
 assert x['prefill_response']['output_ids']==row['output_ids'][:1]
 assert last['meta_info']['finish_reason']['type']=='length'
 assert row['cached_tokens_prefill']==0
controls=[json.loads(s) for s in (F/'controls.jsonl').read_text().splitlines()];assert controls[0]['path']=='/flush_cache'
groups=[json.loads(s) for s in (F/'groups.jsonl').read_text().splitlines()];assert len(groups)==1;g=groups[0]
ss=[]
for s in gzip.open(F/'raw-metrics.jsonl.gz','rt'):
 x=json.loads(s)
 if x['role']!='decode' or not (g['started_monotonic']<=x['started_monotonic']<=x['ended_monotonic']<=g['ended_monotonic']):continue
 raw=x.pop('raw');x['raw_sha256']=hashlib.sha256(raw.encode()).hexdigest()
 x['metrics']={k:metric(raw,k) for k in ['kv_used_tokens','num_running_reqs','decode_sum_seq_lens','kv_available_tokens','max_total_num_tokens']}
 if x['metrics']['kv_used_tokens'] is not None:ss.append(x)
full=[x for x in ss if x['metrics']['num_running_reqs']==32 and x['metrics']['kv_used_tokens']>=1048576 and x['metrics']['decode_sum_seq_lens']>=1048576]
# Preserve the original full-32 gate; report its actual outcome, without relaxing it.
full32_passed=len(full)>=2
active=[x for x in ss if x['metrics']['decode_sum_seq_lens'] is not None]
active_peak=max(active,key=lambda x:x['metrics']['decode_sum_seq_lens'])
peak=max(ss,key=lambda x:x['metrics']['kv_used_tokens'])
report={'passed':True,'raw_integrity_passed':True,'full32_residency_gate_passed':full32_passed,'original_full32_gate_failure_preserved':(R/'1p1d-KV-stress-failed-r182.json').exists(),'max_running_requests_observed':max(x['metrics']['num_running_reqs'] for x in ss),'peak_active_sequence_sample':active_peak,'unix_s':time.time(),'completed_requests':32,'input_tokens_each':32768,'output_tokens_each':2048,'submitted_prompt_tokens':1048576,'eligible_decode_samples':len(ss),'full_prompt_resident_samples':len(full),'same_scrape_peak':peak,'batch_elapsed_s':g['elapsed_s'],'scope':'Raw integrity audit and observed lower bounds. Original full32 simultaneous residency gate is retained and reported separately; passed does not accept that gate. Active sequence length excludes queued reservations; KV used can include queued/cached allocations. No P+D sum, no hardware maximum or task quality.','files_sha256':{n:hashlib.sha256((F/n).read_bytes()).hexdigest() for n in ['complete.json','server-info.json','requests.jsonl','groups.jsonl','controls.jsonl','raw-metrics.jsonl.gz']}}
with (R/'1p1d-KV-observed-audit-r185.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
