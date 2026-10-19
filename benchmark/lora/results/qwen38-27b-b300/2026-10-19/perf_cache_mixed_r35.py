"""Natural multi-adapter SSE benchmark; explicit cold/warm projection-cache tradeoff."""
import argparse,concurrent.futures,json,threading,time,urllib.request
from pathlib import Path
from compare_merged import call,quantile
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'server-info.json').write_text(json.dumps(call(a.url,'/get_server_info'),indent=2))
adapters={'real':'/work/qwen38-mol-experiment/adapters/real_r16','half':'/work/qwen38-mol-merged-bf16-20261019/adapters/real_half','gdn':'/work/qwen38-mol-experiment/adapters/gdn_fused'}
api=[]
def control(path,data):
 t=time.monotonic();res=call(a.url,path,data);api.append({'path':path,'request':data,'response':res,'elapsed_s':time.monotonic()-t});(a.output/'api.json').write_text(json.dumps(api,indent=2));return res
loaded=[]
try:
 for name,path in adapters.items():control('/load_lora_adapter',{'lora_name':'mix-'+name,'lora_path':path});loaded.append('mix-'+name)
 corpus=json.loads(Path('/work/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text());tokens=max(corpus,key=lambda x:len(x['input_ids']))['input_ids'];prompts=[]
 for i in range(32):
  ids=(tokens*((2048+len(tokens)-1)//len(tokens)))[:2048];ids[1]=1000+i;prompts.append((['base','real','half','gdn'][i%4],ids))
 def stream(i,kind,ids,label,barrier):
  req={'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':256,'ignore_eos':True}}
  if kind!='base':req['lora_path']='mix-'+kind
  barrier.wait();t=time.monotonic();first=last=None;first_count=prev=0;events=[];final=None
  q=urllib.request.Request(a.url+'/generate',json.dumps(req).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(q,timeout=3600) as f:
   for line in f:
    line=line.decode().strip()
    if not line.startswith('data:'):continue
    raw=line[5:].strip()
    if raw=='[DONE]':break
    final=json.loads(raw);now=time.monotonic();events.append({'elapsed_s':now-t,'data':final});n=len(final.get('output_ids',[]))
    if n>prev:
     if first is None:first,first_count=now,n
     last=now;prev=n
  elapsed=time.monotonic()-t
  (a.output/(label+f'-i{i}.json')).write_text(json.dumps({'request':req,'events':events}))
  assert first is not None and prev==256
  return {'label':label+f'-i{i}','identity':kind,'output_ids':final['output_ids'],'e2e_s':elapsed,'ttft_s':first-t,'first_event_tokens':first_count,'tpot_s':(last-first)/(256-first_count),'cached_tokens':final['meta_info'].get('cached_tokens')}
 def batch(label):
  barrier=threading.Barrier(32);t=time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:rows=list(ex.map(lambda x:stream(x[0],*x[1],label,barrier),enumerate(prompts)))
  return rows,time.monotonic()-t
 warm=[]
 for repeat in range(2):
  control('/flush_cache',{});rows,elapsed=batch(f'warm-r{repeat}');warm.append({'round':repeat,'elapsed_s':elapsed,'output_throughput_tok_s':8192/elapsed});(a.output/'warmup.json').write_text(json.dumps(warm,indent=2))
 rows_all=[];groups=[]
 for repeat in range(3):
  for cache in ['cold','hot']:
   if cache=='cold':control('/flush_cache',{})
   rows,elapsed=batch(f'r{repeat}-{cache}')
   for row in rows:row.update(round=repeat,cache=cache)
   rows_all+=rows;g={'round':repeat,'cache':cache,'elapsed_s':elapsed,'output_throughput_tok_s':8192/elapsed};groups.append(g)
   with (a.output/'requests.jsonl').open('a') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
   with (a.output/'groups.jsonl').open('a') as f:f.write(json.dumps(g)+'\n')
   print(json.dumps(g),flush=True)
 assert len(rows_all)==192
 points=[]
 for cache in ['cold','hot']:
  rows=[x for x in rows_all if x['cache']==cache];gs=[x for x in groups if x['cache']==cache]
  point={'cache':cache,'requests':len(rows),'output_throughput_tok_s':sum(x['output_throughput_tok_s'] for x in gs)/3}
  for metric in ['e2e_s','ttft_s','tpot_s']:point[metric]={f'p{int(q*100)}':quantile([x[metric] for x in rows],q) for q in [.5,.95,.99]}
  points.append(point)
 (a.output/'summary.json').write_text(json.dumps({'request_count':192,'groups':6,'input_tokens':2048,'output_tokens':256,'concurrency':32,'adapters_including_base':4,'requests_per_identity_per_batch':8,'warmup_batches':2,'points':points,'scope':'Natural generated outputs, no forcing. Mixed cache can thrash when its projection working set exceeds the budget. API load timing is CPU registration, not isolated GPU reload time.'},indent=2))
finally:
 for name in reversed(loaded):control('/unload_lora_adapter',{'lora_name':name})
