"""Natural greedy SSE latency/throughput for one isolated topology/endpoint."""
import argparse,concurrent.futures,json,sys,threading,time,urllib.request
from pathlib import Path
sys.path.insert(0,'/sgl-workspace/sglang/benchmark/lora')
from compare_merged import call as endpoint_call,quantile
from pd_native import NativePDClient
import copy,uuid
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter-path');p.add_argument('--profile',action='store_true');p.add_argument('--prefill-url',required=True);p.add_argument('--bootstrap-port',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
a.output.mkdir(exist_ok=False,parents=True)
transport=NativePDClient(a.prefill_url,a.url,a.bootstrap_port,a.output)
call=transport.call
(a.output/'prefill-server-info.json').write_text(json.dumps(endpoint_call(a.prefill_url,'/get_server_info'),indent=2))
for role,url in [('prefill',a.prefill_url),('decode',a.url)]:
 (a.output/(role+'-metrics-before.txt')).write_text(endpoint_call(url,'/metrics'))
info=call(a.url,'/get_server_info');(a.output/'server-info.json').write_text(json.dumps(info,indent=2))
corpus=json.loads(Path('/work/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text());tokens=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
if a.adapter_path:call(a.url,'/load_lora_adapter',{'lora_name':'perf-A','lora_path':a.adapter_path,'pinned':info.get('server_args',info).get('attn_dp_size',1)>1})
rows=[];groups=[]
def stream(ids,nout,label,barrier):
 req={'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':nout,'ignore_eos':True},'bootstrap_host':'127.0.0.1','bootstrap_port':a.bootstrap_port,'bootstrap_room':uuid.uuid4().int & ((1<<63)-1)}
 if a.adapter_path:req['lora_path']='perf-A'
 preq=copy.deepcopy(req);preq['stream']=False
 def prefill():
  t=time.monotonic();res=endpoint_call(a.prefill_url,'/generate',preq)
  return {'request':preq,'response':res,'http_elapsed_s':time.monotonic()-t}
 barrier.wait();t=time.monotonic();first=last=None;prev=0;first_count=None;events=[];final=None
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
  future=ex.submit(prefill)
  request=urllib.request.Request(a.url+'/generate',json.dumps(req).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=3600) as f:
   for line in f:
    line=line.decode().strip()
    if not line.startswith('data:'):continue
    raw=line[5:].strip()
    if raw=='[DONE]':break
    final=json.loads(raw);now=time.monotonic();events.append({'elapsed_s':now-t,'data':final});n=len(final.get('output_ids',[]))
    if n>prev:
     if first is None:first=now;first_count=n
     last=now;prev=n
  prefill_result=future.result()
 elapsed=time.monotonic()-t
 (a.output/(label+'.json')).write_text(json.dumps({'prefill':prefill_result,'decode_request':req,'decode_events':events}))
 assert first is not None and len(final['output_ids'])==nout
 pres=prefill_result['response'];assert len(pres['output_ids'])==1 and pres['output_ids'][0]==final['output_ids'][0]
 assert nout>first_count
 return {'label':label,'input_tokens':len(ids),'output_tokens':nout,'e2e_s':elapsed,'ttft_s':first-t,'tpot_s':(last-first)/(nout-first_count),'first_nonempty_event_tokens':first_count,'prefill_http_elapsed_s':prefill_result['http_elapsed_s'],'cached_tokens':pres['meta_info'].get('cached_tokens'),'output_ids':final['output_ids']}
def batch(prompts,nout,label):
 barrier=threading.Barrier(len(prompts));t=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
  res=list(ex.map(lambda x:stream(x[1],nout,label+f'-i{x[0]}',barrier),enumerate(prompts)))
 return res,time.monotonic()-t
for nin,nout in ([(128,16)] if a.smoke else [(128,128),(2048,256)]):
 for c in ([1] if a.smoke else [1,4,16,32]):
  prompts=[]
  for i in range(c):
   ids=(tokens*((nin+len(tokens)-1)//len(tokens)))[:nin];ids[1]=1000+i;prompts.append(ids)
  for warm in range(1 if a.smoke else 2):
   call(a.url,'/flush_cache',{});batch(prompts,nout,f'warm-in{nin}-c{c}-r{warm}')
  for repeat in range(1 if a.smoke else 3):
   for cache in ['cold','hot']:
    if cache=='cold':call(a.url,'/flush_cache',{})
    label=f'in{nin}-out{nout}-c{c}-r{repeat}-{cache}';res,elapsed=batch(prompts,nout,label)
    for row in res:row.update({'concurrency':c,'round':repeat,'cache':cache})
    rows.extend(res);g={'input_tokens':nin,'output_tokens':nout,'concurrency':c,'round':repeat,'cache':cache,'elapsed_s':elapsed,'output_throughput_tokens_s':c*nout/elapsed};groups.append(g)
    with (a.output/'requests.jsonl').open('a') as f:
     for row in res:f.write(json.dumps(row)+'\n')
    with (a.output/'groups.jsonl').open('a') as f:f.write(json.dumps(g)+'\n')
    print(json.dumps(g),flush=True)
points=[]
for nin in ([128] if a.smoke else [128,2048]):
 for c in ([1] if a.smoke else [1,4,16,32]):
  for cache in ['cold','hot']:
   rr=[r for r in rows if (r['input_tokens'],r['concurrency'],r['cache'])==(nin,c,cache)]
   gg=[g for g in groups if (g['input_tokens'],g['concurrency'],g['cache'])==(nin,c,cache)]
   v={'input_tokens':nin,'concurrency':c,'cache':cache,'requests':len(rr),'throughput_tokens_s_mean':sum(g['output_throughput_tokens_s'] for g in gg)/(1 if a.smoke else 3)}
   for metric in ['e2e_s','ttft_s','tpot_s']:
    values=[r[metric] for r in rr];v[metric]={'mean':sum(values)/len(values),'p50':quantile(values,.5),'p95':quantile(values,.95),'p99':quantile(values,.99)}
   points.append(v)
assert len(rows)==(2 if a.smoke else 636) and len(groups)==(2 if a.smoke else 48)
(a.output/'summary.json').write_text(json.dumps({'request_count':len(rows),'groups':len(groups),'points':points,'warmup_batches_per_point':1 if a.smoke else 2,'rounds':1 if a.smoke else 3,'scope':'Real native PD pairing; natural SSE, no scorer/forcing. TTFT includes bootstrap and transfer. TPOT uses (last-first)/(nout-first_event_token_count). P HTTP time includes prefill and transfer and is not pure transfer latency. Raw Prometheus counters/histograms are retained separately. Same requests repeat on independent PD M.'},indent=2))
if a.profile:
 call(a.url,'/flush_cache',{})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  futures=[ex.submit(endpoint_call,url,'/start_profile',{'output_dir':str(a.output/'profile'/role),'num_steps':2 if role=='prefill' else 10,'activities':['CPU','GPU'],'record_shapes':False,'with_stack':False}) for role,url in [('prefill',a.prefill_url),('decode',a.url)]]
  time.sleep(1)
  for i in range(3):batch([tokens[:128]],32,f'profile-128-32-{i}')
  results=[f.result() for f in futures]
 (a.output/'profile-result.json').write_text(json.dumps(results))
for role,url in [('prefill',a.prefill_url),('decode',a.url)]:
 (a.output/(role+'-metrics-after.txt')).write_text(endpoint_call(url,'/metrics'))
if a.adapter_path:call(a.url,'/unload_lora_adapter',{'lora_name':'perf-A'})
(a.output/'complete.json').write_text(json.dumps({'requests':len(rows),'profile':a.profile,'smoke':a.smoke}))
