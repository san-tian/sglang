"""Observed first no-cache/LRU/active controls; reverse controls still required."""
import hashlib,json,math,gzip
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
policies=['nocache','lru','active'];points={};records={};counters={};profiles={};proofs={};configs={}
for i,policy in enumerate(policies):
 audit=R/f'mixed-cohort-audit-{i}-{policy}-r93.json';report=read(audit);assert report['passed'] and report['natural_requests']==192
 directory=R/f'perf-active-mix-{i}-{policy}-r84';summ=read(directory/'summary.json');assert sha(directory/'summary.json')==report['summary_sha256']
 cfg=read(directory/'server-info.json');configs[policy]=cfg.get('server_args',cfg)
 record={}
 for name,h in report['original_request_files_sha256'].items():
  path=directory/name;assert sha(path)==h
  raw=read(path);record[name]={'request':raw['request'],'output_ids':raw['events'][-1]['data']['output_ids']}
 assert len(record)==192;records[policy]=record
 points[policy]=report['points'];proofs[policy]={'cohort_audit_sha256':sha(audit),'raw_requests_rehashed':192}
 cfile=R/(directory.name+'-cache-counters.json');counter=read(cfile);before=counter['before'] or {};after=counter['after'];assert after
 keys=['hits','misses','hit_bytes','merged_bytes','admissions','bypasses','evictions','invalidations']
 delta={k:after[k]-before.get(k,0) for k in keys};assert all(v>=0 for v in delta.values())
 assert after['budget_bytes']==(0 if policy=='nocache' else 36864<<20)
 if policy=='nocache':assert after['hits']==after['admissions']==after['peak_retained_bytes']==0
 else:assert after['peak_retained_bytes']<=after['budget_bytes']
 if policy=='active':assert after['transient_reserve_bytes']>0 and after['peak_retained_bytes']<=after['budget_bytes']-after['transient_reserve_bytes']
 assert after['policy']==('lru' if policy=='nocache' else policy)
 counters[policy]={'stage_delta':delta,'snapshot':after,'scope':'Whole client stage includes2warmup batches+192measured requests and adapter load/unload, not only the measured interval.','source_sha256':sha(cfile)}
 pfile=R/f'profile-active-mix-{i}-{policy}-r84/kernel-summary.json';profile=read(pfile);assert len(profile['traces'])==1
 for t in profile['traces']:
  path=Path(t['file']);assert sha(path)==t['sha256'] and t['gpu_kernel_count']>0
  with gzip.open(path,'rt') as f:doc=json.load(f)
  kernels=[e for e in doc['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X' and isinstance(e.get('dur'),(int,float))]
  assert len(kernels)==t['gpu_kernel_count'] and all(math.isfinite(e['dur']) and e['dur']>=0 for e in kernels)
  assert sum(v['count'] for v in t['categories'])==len(kernels)
 profiles[policy]={'source_sha256':sha(pfile),'kernel_events':sum(t['gpu_kernel_count'] for t in profile['traces']),'categories':profile['traces'][0]['categories']}
assert all(records[p]==records['nocache'] for p in policies),'Requests or natural output paths differ across no-cache/LRU controls'
common=['tp_size','attn_dp_size','context_length','dtype','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','page_size','chunked_prefill_size','max_running_requests']
assert all(configs['nocache'][k]==configs['lru'][k] for k in common)
comparisons=[]
for cache in ['cold','hot']:
 c={p:next(v for v in points[p] if v['cache']==cache) for p in policies}
 comparisons.append({'cache_label':cache,'controls':c,'LRU_over_nocache_throughput':c['lru']['throughput_tokens_s_mean']/c['nocache']['throughput_tokens_s_mean'],'active_over_nocache_throughput':c['active']['throughput_tokens_s_mean']/c['nocache']['throughput_tokens_s_mean'],'active_over_LRU_throughput':c['active']['throughput_tokens_s_mean']/c['lru']['throughput_tokens_s_mean']})
result={'passed':True,'native_requests':576,'natural_outputs_identical':True,'identical_request_payloads':True,'comparison':comparisons,'cache_counters':counters,'profiles':profiles,'proofs':proofs,'source_sha256':sha(Path(__file__)),'scope':'First three R84 sequential controls onGPU2, R55 identical image/mem.52/KV65536/SSM256. Largest32*(2048+256)=73728 exceedsKV pool; inspect queue and actual hits. Observed order can influence clocks; reverse-order controls remain required. No final policy win, isolated speedup, multiGPU or4P4D acceptance.'}
out=R/'mixed-first-three-comparison-r98.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ['passed','native_requests','natural_outputs_identical','comparison']}))
