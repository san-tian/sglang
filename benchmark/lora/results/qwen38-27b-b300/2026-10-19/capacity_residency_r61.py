"""Bounded, unique-prefix resident-KV experiment; reports observations, not model maxima."""
import argparse,concurrent.futures,gzip,hashlib,json,re,threading,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter-path',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def request(path,data=None):
 q=urllib.request.Request(a.url+path,None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=1800) as f:
  raw=f.read().decode()
 try:return json.loads(raw)
 except ValueError:return raw
info=request('/get_server_info');(a.output/'server-info.json').write_text(json.dumps(info,indent=2));cfg=info.get('server_args',info)
assert cfg['enable_metrics']
request('/load_lora_adapter',{'lora_name':'capacity-A','lora_path':a.adapter_path,'pinned':cfg['attn_dp_size']>1})
corpus=json.loads(Path('/work/qwen38-mol-speed-20261019/corpus-mixed.json').read_text());filler=max(corpus,key=lambda r:len(r['input_ids']))['input_ids']
metrics=[];stop=threading.Event();errors=[]
names=['num_running_reqs','num_queue_reqs','decode_sum_seq_lens','kv_used_tokens','kv_available_tokens','kv_evictable_tokens','num_retracted_reqs','num_retracted_requests_total','max_total_num_tokens','kv_cache_memory_usage_gb','mamba_used_tokens','mamba_available_tokens','mamba_evictable_tokens','mamba_usage','num_used_tokens','full_token_usage']
regex=re.compile(r'^(sglang:(?:'+('|'.join(names))+r'))(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$')
def observe():
 while not stop.is_set():
  try:
   observed_start=time.monotonic();raw=request('/metrics');values=[]
   for line in raw.splitlines():
    m=regex.match(line)
    if m:
     labels=m.group(2) or ''
     if 'priority="' in labels and 'priority=""' not in labels:continue
     values.append({'metric':m.group(1),'labels':labels,'value':float(m.group(3))})
   observed_end=time.monotonic();row={'monotonic':observed_end,'observation_started_monotonic':observed_start,'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'metrics':values};metrics.append(row)
   with gzip.open(a.output/'raw-metrics.jsonl.gz','at') as f:f.write(json.dumps({'monotonic':observed_end,'observation_started_monotonic':observed_start,'raw_prometheus':raw})+'\n')
   with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  except Exception as e:errors.append(repr(e))
  stop.wait(.25)
t=threading.Thread(target=observe);t.start()
rows=[]
try:
 for count in [1,4,8]:
  request('/flush_cache',{})
  inputs=[]
  for i in range(count):
   ids=(filler*5)[:6000];ids[:8]=[1000+100*i+j for j in range(8)];inputs.append(ids)
  payload={'input_ids':inputs,'lora_path':['capacity-A']*count,'sampling_params':[{'temperature':0,'max_new_tokens':512,'ignore_eos':True}]*count}
  (a.output/f'payload-c{count}.json').write_text(json.dumps(payload));started=time.monotonic();res=request('/generate',payload);ended=time.monotonic()
  (a.output/f'responses-c{count}.json').write_text(json.dumps(res));assert len(res)==count
  assert all(len(x['output_ids'])==512 and x['meta_info']['prompt_tokens']==6000 and x['meta_info']['finish_reason']['type']!='abort' for x in res)
  samples=[r for r in metrics if started<=r['observation_started_monotonic']<=r['monotonic']<=ended]
  assert samples and any(r['metrics'] for r in samples), 'No live metrics; do not infer resident tokens from submitted input lengths'
  def total(sample,name):return sum(v['value'] for v in sample['metrics'] if v['metric']=='sglang:'+name)
  peaks={name:max(total(r,name) for r in samples) for name in names}
  snapshot=max(samples,key=lambda r:total(r,'kv_used_tokens'))
  rows.append({'http_concurrency':count,'submitted_unique_prompt_tokens':6000*count,'output_tokens_per_request':512,'elapsed_s':ended-started,'started_monotonic':started,'ended_monotonic':ended,'metric_peaks':peaks,'peak_residency_snapshot':snapshot})
finally:
 stop.set();t.join();request('/unload_lora_adapter',{'lora_name':'capacity-A'})
assert request('/health')==''
(a.output/'summary.json').write_text(json.dumps({'scope':'Observed lower bounds under a deliberately capped KV byte budget and 16K configured context; not the official per-request limit or the maximum capacity of the entire GPU','cases':rows,'metric_errors':errors,'unique_prefix_tokens':8,'all_outputs_complete':True,'raw_metrics_file':'raw-metrics.jsonl.gz','metric_aggregation_note':'Legacy metric_peaks sums all matching series and is NOT a logical resident token count. Independently normalize actual labels/reporting leaders and use a single same-time snapshot, rather than summing per-worker peaks or TP duplicate constants. kv_cache_memory_usage_gb can include multiple pools; inspect actual runtime allocation logs to separate full-attention KV from SSM.'},indent=2))
print(json.dumps(rows),flush=True)
