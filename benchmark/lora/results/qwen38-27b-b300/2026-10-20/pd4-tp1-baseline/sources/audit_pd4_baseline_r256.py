"""Recompute baseline results from raw SSE and time-aligned per-worker samples."""
import collections,gzip,hashlib,json,math,re,statistics,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def percentile(a,p):
 a=sorted(a);return a[math.ceil(p*len(a))-1] if a else None
def stats(a):return {'count':len(a),'mean':statistics.mean(a),'p50':percentile(a,.5),'p95':percentile(a,.95),'p99':percentile(a,.99),'max':max(a)} if a else {'count':0}
def metric(raw,name):
 for line in raw.splitlines():
  if not line.startswith('sglang:'+name+'{'):continue
  left,v=line.rsplit(' ',1);labels=dict(re.findall(r'(\w+)="([^\"]*)"',left))
  if labels.get('priority','')=='' and labels.get('tp_rank','0')=='0':return float(v)
 return None
report={'unix_s':time.time(),'phases':{},'topology':json.loads((E/'topology.json').read_text()),'precision_scope':'replica/PD consistency only; independent FP8 merge acceptance remains incomplete; previous MTP precision gate remains failed'}
for phase in ['smoke','speed','natural','capacity','slots']:
 f=E/('capacity-r263' if phase=='capacity' and (E/'capacity-r263/complete.json').exists() else phase)
 if not (f/'complete.json').exists():continue
 reqs=rows(f/'requests.jsonl');groups=rows(f/'groups.jsonl')
 for row in reqs:
  with gzip.open(f/(row['label']+'.json.gz'),'rt') as h:raw=json.load(h)
  q=raw['request'];ev=raw['events'];last=ev[-1]['data'];pf=raw['prefill_response']
  assert raw['error'] is None and len(q['input_ids'])==row['input_tokens']
  assert last['output_ids']==row['output_ids'] and len(last['output_ids'])==row['output_tokens']
  assert pf['output_ids']==last['output_ids'][:1]
  assert q['bootstrap_room']==raw['prefill_request']['bootstrap_room'] and q['bootstrap_port']==raw['p']['bootstrap_port']
  if q['sampling_params']['ignore_eos']:assert row['output_tokens']==q['sampling_params']['max_new_tokens'] and last['meta_info']['finish_reason']['type']=='length'
  first=next(x for x in ev if x['data'].get('output_ids'))
  assert math.isclose(row['ttft_s'],first['elapsed_s'],abs_tol=1e-7)
  if row['tpot_s'] is not None:
   want=(ev[-1]['elapsed_s']-first['elapsed_s'])/(row['output_tokens']-row['first_event_tokens'])
   # Last SSE may repeat the final output; use the first event that reaches final length.
   last_new=next(x for x in ev if len(x['data'].get('output_ids',[]))==row['output_tokens'])
   want=(last_new['elapsed_s']-first['elapsed_s'])/(row['output_tokens']-row['first_event_tokens'])
   assert math.isclose(row['tpot_s'],want,abs_tol=1e-7)
 for g in groups:
  rs=[r for r in reqs if r['label'].startswith(g['label']+'-i')]
  assert len(rs)==g['concurrency'] and sum(r['output_tokens'] for r in rs)==g['output_tokens']
  assert math.isclose(g['output_tokens_s'],g['output_tokens']/g['elapsed_s'],rel_tol=1e-12)
  assert g['p_counts']==[sum(r['p']==i for r in rs) for i in range(4)]
  assert g['d_counts']==[sum(r['d']==i for r in rs) for i in range(4)]
 info=json.loads((f/'server-info.json').read_text())
 rp={'verified_requests':len(reqs),'complete':True,'kv_tokens_per_worker':{n:i['max_total_num_tokens'] for n,i in info.items()},'groups':groups}
 samples=[]
 for s in rows(f/'gpu-metrics.jsonl'):
  gpus={}
  for line in s['raw'].splitlines():
   a=[float(x.strip()) for x in line.split(',')];gpus[int(a[0])]={'memory_used_mib':a[1],'gpu_util_pct':a[2],'memory_util_pct':a[3],'power_w':a[4],'temperature_c':a[5]}
  samples.append({'t':s['monotonic'],'gpus':gpus})
 ms=[]
 if (f/'raw-metrics.jsonl.gz').exists():
  with gzip.open(f/'raw-metrics.jsonl.gz','rt') as h:
   for line in h:
    s=json.loads(line)
    if 'raw' not in s:continue
    raw=s.pop('raw');s['metrics']={k:metric(raw,k) for k in ['kv_used_tokens','num_running_reqs','num_queue_reqs','decode_sum_seq_lens','max_total_num_tokens']};ms.append(s)
 if phase in ['speed','natural']:
  configs=collections.defaultdict(list)
  for g in groups:
   if not g['label'].startswith('warm-'):configs[(str(g['input_tokens']),g['max_output_tokens'],g['concurrency'])].append(g)
  rp['speed']=[]
  for (nin,nout,c),gs in configs.items():
   rs=[r for r in reqs if any(r['label'].startswith(g['label']+'-i') for g in gs)]
   gpu=[s for s in samples if any(g['started_monotonic']<=s['t']<=g['ended_monotonic'] for g in gs)]
   qm=[s for s in ms if any(g['started_monotonic']<=s['started_monotonic']<=s['ended_monotonic']<=g['ended_monotonic'] for g in gs)]
   meta=[r['meta_info'] for r in rs];nv=sum(m.get('spec_verify_ct',0) for m in meta);prop=sum(m.get('spec_num_proposed_drafts',0) for m in meta);corr=sum(m.get('spec_num_correct_drafts',0) for m in meta)
   rp['speed'].append({'input_tokens':nin,'max_output_tokens':nout,'concurrency':c,'rounds':len(gs),'requests':len(rs),
    'throughput_rounds':[g['output_tokens_s'] for g in gs],'throughput_mean':statistics.mean(g['output_tokens_s'] for g in gs),
    'ttft_s':stats([r['ttft_s'] for r in rs]),'tpot_s':stats([r['tpot_s'] for r in rs if r['tpot_s'] is not None]),'e2e_s':stats([r['e2e_s'] for r in rs]),
    'input_tokens_s_mean':statistics.mean(g['input_tokens_s'] for g in gs),'actual_output_tokens':sum(r['output_tokens'] for r in rs),
    'accept_rate':corr/prop if prop else None,'accept_length':sum(r['output_tokens'] for r in rs)/nv if nv else None,
    'gpu_util_pct_by_gpu':{str(i):stats([s['gpus'][i]['gpu_util_pct'] for s in gpu]) for i in range(8)},
    'worker_max_running':{w['name']:max([s['metrics']['num_running_reqs'] or 0 for s in qm if s['worker']==w['name']],default=None) for w in report['topology']['workers']}})
 if phase=='capacity':
  rp['residency']=[]
  for g in groups:
   eligible=[s for s in ms if s['role']=='decode' and g['started_monotonic']<=s['started_monotonic']<=s['ended_monotonic']<=g['ended_monotonic'] and s['metrics']['num_running_reqs'] is not None]
   # Samples are concurrent HTTP polls, not one atomic global scrape. Preserve spans.
   windows=[]
   for start in range(0,len(ms),8):
    ss=ms[start:start+8];ds=[s for s in ss if s['role']=='decode']
    if len(ss)!=8 or len(ds)!=4 or len({s['index'] for s in ds})!=4:continue
    t0=min(s['started_monotonic'] for s in ds);t1=max(s['ended_monotonic'] for s in ds)
    if not g['started_monotonic']<=t0<=t1<=g['ended_monotonic'] or t1-t0>1:continue
    if any(s['metrics']['decode_sum_seq_lens'] is None for s in ds):continue
    windows.append({'started_monotonic':t0,'ended_monotonic':t1,'span_s':t1-t0,'running_sum':sum(s['metrics']['num_running_reqs'] or 0 for s in ds),'active_seq_sum':sum(s['metrics']['decode_sum_seq_lens'] for s in ds),'kv_used_sum':sum(s['metrics']['kv_used_tokens'] or 0 for s in ds),'workers':ds})
   rp['residency'].append({'label':g['label'],'concurrency':g['concurrency'],'input_tokens':g['input_tokens'],'configured_total_each':g['input_tokens']+g['max_output_tokens'],'elapsed_s':g['elapsed_s'],
    'max_running_by_d':{str(i):max([s['metrics']['num_running_reqs'] or 0 for s in eligible if s['index']==i],default=None) for i in range(4)},
    'peak_near_concurrent_window':max(windows,key=lambda s:s['active_seq_sum']) if windows else None,'scope':'natural unpaced residency; four concurrent polls within <=1s, never sum independent per-worker peaks'})
 if phase=='smoke':
  pair=[r for r in reqs if r['label'].startswith('pair-')]
  rp['pair_coverage']={str(a):sorted({(r['p'],r['d']) for r in pair if r['adapter']==a}) for a in [None,'main']}
  rp['replica_consistency']=[]
  for prefix in ['pair-None-','pair-main-']+[f'natural-i{i}-' for i in range(8)]:
   rs=[r for r in reqs if r['label'].startswith(prefix)];ref=rs[0];diffs=[];first=[];decode=[]
   for r in rs[1:]:
    for a,b in zip(r['prefill_meta']['input_token_logprobs'],ref['prefill_meta']['input_token_logprobs']):
     if a[0] is not None and b[0] is not None:diffs.append(abs(a[0]-b[0]))
    a,b=r['meta_info']['output_token_logprobs'],ref['meta_info']['output_token_logprobs']
    first.append(abs(a[0][0]-b[0][0]));lcp=next((i for i,(x,y) in enumerate(zip(r['output_ids'],ref['output_ids'])) if x!=y),len(r['output_ids']))
    decode.extend(abs(x[0]-y[0]) for x,y in zip(a[1:lcp],b[1:lcp]))
   rp['replica_consistency'].append({'prefix':prefix,'requests':len(rs),'whole_output_equal_to_reference':sum(r['output_ids']==ref['output_ids'] for r in rs),'prefill_abs_nat':stats(diffs),'first_abs_nat':stats(first),'decode_common_prefix_abs_nat':stats(decode)})
 if phase=='slots':
  models=json.loads((f/'models-loaded32.json').read_text());rp['models_loaded32']={n:[x['id'] for x in v['data']] for n,v in models.items()}
  refs=json.loads((f/'slot-references.json').read_text());rp['mixed_vs_single']=[{k:r[k] for k in ['identity','d','mixed_label','same_output_ids']} for r in refs]
  mixed=[r for r in reqs if r['label'].startswith('slots32-eachD-c128-i')]
  rp['unique_adapters_seen_by_d']={str(i):sorted({r['adapter'] for r in mixed if r['d']==i}) for i in range(4)}
 report['phases'][phase]=rp
report['verified_total_requests']=sum(p['verified_requests'] for p in report['phases'].values())
report['all_five_phases_complete']=len(report['phases'])==5
report['raw_files_sha256']={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for phase in report['phases'] for p in (E/('capacity-r263' if phase=='capacity' and (E/'capacity-r263/complete.json').exists() else phase)).rglob('*') if p.is_file()}
(E/'audit-r256.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'verified_total_requests':report['verified_total_requests'],'phases':list(report['phases']),'all_five_phases_complete':report['all_five_phases_complete']}))
