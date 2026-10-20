"""Independent raw residency and boundary audit, separate from throughput statistics."""
import gzip,hashlib,json,math,re,statistics,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253';F=E/'residency-paced'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def metric(raw,n):
 for line in raw.splitlines():
  if line.startswith('sglang:'+n+'{'):
   labels=dict(re.findall(r'(\w+)="([^\"]*)"',line.rsplit(' ',1)[0]))
   if labels.get('priority','')=='' and labels.get('tp_rank','0')=='0':return float(line.rsplit(' ',1)[1])
 return None
def verify(folder):
 rs=rows(folder/'requests.jsonl')
 for r in rs:
  with gzip.open(folder/(r['label']+'.json.gz'),'rt') as f:raw=json.load(f)
  last=raw['events'][-1]['data'];assert raw['error'] is None
  assert last['output_ids']==r['output_ids'] and len(last['output_ids'])==r['output_tokens']
  assert len(raw['request']['input_ids'])==r['input_tokens']
  assert raw['prefill_response']['output_ids']==last['output_ids'][:1]
  assert raw['prefill_response']['meta_info']==r['prefill_meta'] and last['meta_info']==r['meta_info']
 return rs
assert json.loads((F/'complete.json').read_text())['passed']
rs=verify(F);assert len(rs)==128 and all(r['input_tokens']==131072 and r['output_tokens']==512 for r in rs)
assert all(sum(r['d']==i for r in rs)==32 for i in range(4))
qualified=[]
for w in rows(F/'gate-metrics.jsonl'):
 ss=w['samples'];assert len(ss)==4 and {s['index'] for s in ss}==set(range(4))
 vals=[]
 for s in ss:
  m={n:metric(s['raw'],n) for n in ['num_running_reqs','decode_sum_seq_lens','kv_used_tokens','max_total_num_tokens']}
  assert m==s['metrics'];vals.append(m)
 span=max(s['ended_monotonic'] for s in ss)-min(s['started_monotonic'] for s in ss)
 assert abs(span-w['span_s'])<1e-6
 if span<=1 and all(m['num_running_reqs']==32 and m['decode_sum_seq_lens']>=32*131072 for m in vals):
  qualified.append({'monotonic':w['monotonic'],'span_s':span,'running_requests':sum(m['num_running_reqs'] for m in vals),'active_seq_tokens':sum(m['decode_sum_seq_lens'] for m in vals),'kv_used_tokens':sum(m['kv_used_tokens'] for m in vals),'per_d':vals})
assert len(qualified)>=2
controls=rows(F/'pacing-controls.jsonl');assert controls[0]['forward_sleep_time']==4
resets=[c for c in controls if c['forward_sleep_time'] is None];assert len(resets)>=2 and all(len(c['responses'])==4 for c in controls)
old=verify(E/'capacity');assert len(old)==8
failed=[]
for p in (E/'capacity').glob('capacity-in262016-*.json.gz'):
 with gzip.open(p,'rt') as f:x=json.load(f)
 assert x['error'] and any('262148' in json.dumps(e['data']) for e in x['events']),str(p)
 failed.append(str(p.relative_to(E)))
assert len(failed)==4
boundary=json.loads((E/'capacity-r263/boundary-rejections.json').read_text());assert len(boundary['responses'])==8
errors=rows(E/'capacity-r263/http-errors.jsonl')
for w in boundary['responses']:
 if 'response' in w:body=json.dumps(w['response'])
 else:
  ix=next(z for z in json.loads((E/'topology.json').read_text())['workers'] if z['name']==w['worker'])
  body=next(z['body'] for z in errors if z['url']==ix['url']+'/generate')
 assert '262145' in body and '262144' in body,(w['worker'],body)
focused=verify(E/'slots-precision-r266');assert len(focused)==208
by={r['label']:r for r in focused};comparison=json.loads((E/'slots-precision-r266/comparison.json').read_text());ds=[];ps=[];whole=0
for c in comparison['comparisons']:
 a,b=by[c['mixed_label']],by[c['single_label']]
 lcp=next((i for i,(x,y) in enumerate(zip(a['output_ids'],b['output_ids'])) if x!=y),32)
 assert lcp==c['common_prefix_tokens'];whole+=lcp==32
 ds.extend(abs(x[0]-y[0]) for x,y in zip(a['meta_info']['output_token_logprobs'][1:lcp],b['meta_info']['output_token_logprobs'][1:lcp]))
 ps.extend(abs(x[0]-y[0]) for x,y in zip(a['prefill_meta']['input_token_logprobs'],b['prefill_meta']['input_token_logprobs']) if x[0] is not None and y[0] is not None)
for values,key in [(ds,'decode_common_prefix'),(ps,'prefill')]:
 values.sort();assert comparison[key]['count']==len(values)
 assert math.isclose(comparison[key]['mean_abs_nat'],statistics.mean(values),abs_tol=1e-12)
 assert comparison[key]['p99_abs_nat']==values[math.ceil(.99*len(values))-1] and comparison[key]['max_abs_nat']==max(values)
assert comparison['whole_output_equal']==whole
report={'unix_s':time.time(),'passed':True,'verified_paced_requests':len(rs),'verified_precision_replay_requests':len(focused),'precision_replay_independently_recomputed':True,'expected_residency_per_d':32,'input_tokens_each':131072,'qualified_windows':qualified,
 'peak_qualified_window':max(qualified,key=lambda w:w['active_seq_tokens']),'pacing_reset_calls':len(resets),'original_capacity_successes_verified':len(old),'original_expected_rejections_verified':failed,
 'all_8_workers_reject_262141_user_tokens':True,'boundary_configured_context':262144,'mtp_reserved_tokens':4,
 'scope':'Paced capacity proof only; no throughput claim. Four concurrent per-worker scrapes, <=1 second span, not one atomic global sample.',
 'files_sha256':{str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [F,E/'capacity',E/'capacity-r263',E/'slots-precision-r266'] for p in folder.rglob('*') if p.is_file()}}
(E/'residency-audit-r265.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['passed','verified_paced_requests','pacing_reset_calls']}))
