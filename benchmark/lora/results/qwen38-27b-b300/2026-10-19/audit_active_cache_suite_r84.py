"""Audit recorded payloads, native output paths, sealed scores, counters and traces."""
import collections,gzip,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
IMAGE='sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def config(p):
 x=read(p);return x.get('server_args',x)
def gate(s):
 assert s['count']>0 and s['passes_logprob_gate'] is True
 assert all(math.isfinite(s[k]) for k in ['mean_abs_nat','p99_abs_nat','max_abs_nat'])
 assert s['mean_abs_nat']<=.01 and s['p99_abs_nat']<=.1
common=['tp_size','attn_dp_size','context_length','dtype','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','page_size','chunked_prefill_size','max_running_requests']
raw_hashes={};configs={};natural={};summaries={}
def native(path,count):
 cfg=config(path/'server-info.json');configs[path.name]=cfg
 assert cfg['tp_size']==cfg['attn_dp_size']==1
 assert cfg['max_total_num_tokens']==65536 and cfg['max_mamba_cache_size']==256 and cfg['mem_fraction_static']==.52
 rows=lines(path/'requests.jsonl');assert len(rows)==count and len({r['label'] for r in rows})==count
 summary=read(path/'summary.json');assert summary['request_count']==count
 groups=lines(path/'groups.jsonl');assert len(groups)==(48 if count==636 else 6)
 assert len(summary['points'])==(16 if count==636 else 2)
 records={};hashes={}
 for row in rows:
  label=row['label'];p=path/(label+'.json');raw=read(p);req=raw['request'];events=raw['events'];last=events[-1]['data'];nout=row.get('output_tokens',256);nin=row.get('input_tokens',2048)
  assert len(req['input_ids'])==nin
  assert req['sampling_params']=={'temperature':0,'max_new_tokens':nout,'ignore_eos':True}
  assert req['stream'] is True and 'custom_logit_processor' not in req
  assert len(last['output_ids'])==nout and last['output_ids']==row['output_ids']
  assert last['meta_info']['finish_reason']=={'type':'length','length':nout}
  assert last['meta_info']['prompt_tokens']==nin and last['meta_info']['completion_tokens']==nout
  assert len(events[0]['data']['output_ids'])==1
  assert row['ttft_s']==events[0]['elapsed_s'] and row['e2e_s']>=events[-1]['elapsed_s']
  assert all(math.isfinite(row[k]) and row[k]>0 for k in ['ttft_s','e2e_s','tpot_s'])
  if count==192:
   identity=row['identity'];assert req.get('lora_path')==({'base':None,'real':'mix-real','half':'mix-half','gdn':'mix-gdn'}[identity])
  records[label]={'request':req,'output_ids':row['output_ids']};hashes[label]=sha(p)
 for group in groups:
  subset=[r for r in rows if (r['round'],r['cache'])==(group['round'],group['cache']) and (count==192 or (r['input_tokens'],r['concurrency'])==(group['input_tokens'],group['concurrency']))]
  assert len(subset)==(32 if count==192 else group['concurrency'])
  throughput=sum(r.get('output_tokens',256) for r in subset)/group['elapsed_s']
  key='output_throughput_tok_s' if count==192 else 'output_throughput_tokens_s'
  assert abs(throughput-group[key])<1e-8
  if count==192:assert collections.Counter(r['identity'] for r in subset)=={'base':8,'real':8,'half':8,'gdn':8}
 if count==192:
  assert collections.Counter(r['identity'] for r in rows)=={'base':48,'real':48,'half':48,'gdn':48}
  assert len(read(path/'warmup.json'))==2
 raw_hashes[path.name]=hashes;summaries[path.name]=summary;natural[path.name]=records
 return records
single={}
for side in ['m','base','d']:
 name=f'perf-active-cache-r56-{side}-r30';single[side]=native(R/name,636)
 assert config(R/name/'server-info.json')['enable_lora']==(side=='d')
assert set(single['m'])==set(single['d'])==set(single['base'])
for label in single['m']:
 m,d,b=[single[k][label] for k in ['m','d','base']]
 assert m['output_ids']==d['output_ids'],label
 assert m['request']==b['request']
 assert {k:v for k,v in d['request'].items() if k!='lora_path'}==m['request'] and d['request']['lora_path']=='perf-A'
for cfg in configs.values():assert all(cfg[k]==next(iter(configs.values()))[k] for k in common)
accuracy=read(R/'active-cache-r56-independent-audit-r56.json')
assert accuracy['performance']['accepted'] and sum(v['case_pairs'] for v in accuracy['accuracy'].values())==248
for v in accuracy['accuracy'].values():
 for score in v['scores'].values():gate(score)
# Rehash each independent reference again, rather than trusting its audit flag.
for label in ['full','gdn']:
 p=R/f'active-cache-r56-{label}-reference-r30'
 for name,h in read(p/'seal.json')['files'].items():assert sha(p/name)==h
prefix=R/'cache-prefix-dynamic-r56';ps=read(prefix/'summary.json');assert ps['case_pairs']==10 and ps['scores']['count']==160;gate(ps['scores'])
refs={}
for variant in ['real','base','gdn']:
 p=R/'cache-prefix-reference-r56'/variant
 for name,h in read(p/'seal.json')['files'].items():assert sha(p/name)==h
 refs[variant]=p
for row in ps['rows']:
 key,mode=row['variant'],row['cache'];assert row['cached_tokens']==(1600 if mode=='hot' else 0)
 reference={'real':'real','twin':'real','gdn':'gdn','zero':'base','base':'base'}[key]
 response=read(prefix/f'{key}-{mode}-response.json');cont=read(refs[reference]/'natural.json')['output_ids'];assert response['output_ids']==cont
 actual=read(prefix/f'{key}-{mode}-scores.json');expected=read(refs[reference]/f'{reference}-{mode}-scores.json')
 assert actual==expected and len(actual)==16
 cap=lines(prefix/f'{key}-{mode}-capture.rank-0')
 assert [v['step'] for v in cap]==list(range(16)) and [v['token'] for v in cap]==cont
 assert [[v['logprob'],v['argmax']] for v in cap]==actual
 gate(row['scores']);assert row['scores']['max_abs_nat']==0
 if mode=='hot':assert row['cold_hot_noise']['max_abs_nat']==0
mixed=read(R/'active-mixed-r56/summary.json');assert mixed['mixed_requests']==48 and mixed['adapters_including_base']==4 and len(mixed['cases'])==48
assert sum(v['count'] for v in mixed['scores'].values())==768
for row in mixed['cases']:
 for phase in ['prefill_last','decode']:gate(row[phase])
life=read(R/'active-lifecycle-r56/summary.json')
assert life['passed'] and life['serial_cycles']==life['invalid_loads']==100 and life['lru_requests']==108 and life['lru_aliases']==12 and life['inflight_output_tokens']==256
assert life['new_requests_rejected_while_draining'] and life['max_repeat_logprob_abs_nat']==0
api=lines(R/'active-lifecycle-r56/api.jsonl');success=0;rejected=0
for row in api:
 if row['path']=='/generate' and row['status']==200:
  n=row['request']['sampling_params']['max_new_tokens'];body=row['body'];assert len(body['output_ids'])==n
  lp=body['meta_info']['output_token_logprobs'];assert len(lp)==n and all(math.isfinite(p[0]) for p in lp);assert [p[1] for p in lp]==body['output_ids'];success+=1
 if row['status']==400:rejected+=1
assert success>=3+100+100+108 and rejected>=100+12
policies=['nocache','lru','active','active','lru','nocache'];counterproof={};points=[]
for i,policy in enumerate(policies):
 key=f'perf-active-mix-{i}-{policy}-r84';records=native(R/key,192);cfg=configs[key]
 assert cfg['enable_lora'] and cfg['lora_merged_bf16_cache_size_mb']==(0 if policy=='nocache' else 36864)
 assert cfg['lora_merged_bf16_cache_policy']==('lru' if policy=='nocache' else policy)
 assert all(cfg[k]==configs['perf-active-cache-r56-d-r30'][k] for k in common)
 launch=read(R/f'launch-qwen38-active-mix-{i}-{policy}-r84.json');assert IMAGE in launch
 if i:
  assert records==natural['perf-active-mix-0-nocache-r84'],key
 counters=read(R/(key+'-cache-counters.json'));after=counters['after'];before=counters['before'] or {};assert after is not None
 delta={k:after[k]-before.get(k,0) for k in ['hits','misses','hit_bytes','merged_bytes','admissions','bypasses','evictions','invalidations']};assert all(v>=0 for v in delta.values())
 assert after['policy']==('lru' if policy=='nocache' else policy)
 assert after['budget_bytes']==(0 if policy=='nocache' else 36864<<20)
 if policy=='active':assert after['peak_retained_bytes']<=after['budget_bytes']-after['transient_reserve_bytes'] and after['transient_reserve_bytes']>0
 if policy=='nocache':assert after['hits']==after['admissions']==after['peak_retained_bytes']==0
 counterproof[key]={'delta':delta,'snapshot':after,'sha256':sha(R/(key+'-cache-counters.json'))}
 for point in summaries[key]['points']:points.append({'run':i,'policy':policy,**point})
profiles={}
for side in ['m','base','d']:
 p=R/f'perf-active-cache-r56-{side}-r30';summary=read(p/'kernel-summary.json');assert len(summary['traces'])==1
 for trace in summary['traces']:
  file=Path(trace['file']);assert sha(file)==trace['sha256'] and trace['gpu_kernel_count']>0
 profiles['single-'+side]=summary
for i,policy in enumerate(policies[:3]):
 p=R/f'profile-active-mix-{i}-{policy}-r84';summary=read(p/'kernel-summary.json');assert len(summary['traces'])==1
 for trace in summary['traces']:
  file=Path(trace['file']);assert sha(file)==trace['sha256'];opener=gzip.open if file.suffix=='.gz' else open
  with opener(file,'rt') as f:doc=json.load(f)
  kernels=[e for e in doc['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X' and isinstance(e.get('dur'),(int,float))]
  assert len(kernels)==trace['gpu_kernel_count']>0
 profiles[policy]=summary
result={'passed':True,'image':IMAGE,'accuracy':accuracy['accuracy'],'single_native_requests_per_side':636,'mixed_native_requests_total':1152,'native_outputs_identical_across_six_mixed_runs':True,'independent_prefix_pairs':10,'prefix_scores':160,'actual_hot_cached_tokens_per_identity':1600,'lifecycle':life,'lifecycle_api_successful_generations_audited':success,'lifecycle_api_rejections':rejected,'mixed_accuracy_requests':48,'mixed_accuracy_scores':768,'policy_points':points,'projection_cache_counters':counterproof,'mixed_profiles':profiles,'raw_native_request_sha256':raw_hashes,'scope':'R55 GPU2 TP1 fixed .52 memory, 65536 KV rows, 256 Mamba slots. Other GPUs may run background experiments. Correctness gates do not imply a performance benefit. No claim of measured bandwidth or final isolated-machine speedup.'}
(R/'active-cache-suite-audit-r84.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k in ['passed','image','mixed_native_requests_total','native_outputs_identical_across_six_mixed_runs','policy_points']}))
