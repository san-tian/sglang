"""Native FP8/MTP measurements; raw SSE, HTTP failures, per-role metrics retained."""
import concurrent.futures as cf,copy,gzip,json,threading,time,traceback,urllib.request,urllib.error,uuid,subprocess,sys
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');KIND,PHASE=sys.argv[1:3];assert KIND in ['colocated','pd']
ROOT=R/'fp8-mtp-r228';OUT=ROOT/KIND/PHASE;OUT.mkdir(parents=True,exist_ok=False)
URLS={'colocated':'http://127.0.0.1:35800'} if KIND=='colocated' else {'prefill':'http://127.0.0.1:36800','decode':'http://127.0.0.1:37800'}
TARGET=URLS.get('colocated',URLS.get('decode'));STOP=threading.Event();MON=None
AD='/work/qwen38-mol-experiment/adapters/real_r16'
corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text());FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,ensure_ascii=False)
def append(name,value):
 with (OUT/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
def call(url,path,data=None,timeout=1200):
 q=urllib.request.Request(url+path,None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(q,timeout=timeout) as f:raw=f.read().decode()
 except urllib.error.HTTPError as e:
  body=e.read().decode();append('http-errors.jsonl',{'url':url+path,'status':e.code,'body':body,'unix_s':time.time()});raise RuntimeError((url+path,e.code,body)) from e
 try:return json.loads(raw)
 except ValueError:return raw
def control(path,data=None):
 with cf.ThreadPoolExecutor(max_workers=len(URLS)) as pool:values=list(pool.map(lambda u:call(u,path,data),URLS.values()))
 append('controls.jsonl',{'unix_s':time.time(),'path':path,'request':data,'responses':values})
 if path=='/load_lora_adapter':assert all(v.get('success') for v in values),values
 return values
def prompt(n,i):
 ids=(FILLER*((n+len(FILLER)-1)//len(FILLER)))[:n];ids[:8]=[1000+64*i+j for j in range(8)];return ids
def stream(ids,nout,adapter,label,barrier=None,logprobs=False):
 req={'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':nout,'ignore_eos':True}}
 if adapter is not None:req['lora_path']=adapter
 if logprobs:req['return_logprob']=True
 if KIND=='pd':req.update(bootstrap_host='127.0.0.1',bootstrap_port=36810,bootstrap_room=uuid.uuid4().int&((1<<63)-1))
 preq=copy.deepcopy(req);preq['stream']=False
 if barrier:barrier.wait()
 started=time.monotonic();first=last=None;first_count=prev=0;events=[];gaps=[];pool=cf.ThreadPoolExecutor(max_workers=1)
 try:
  pf=pool.submit(call,URLS['prefill'],'/generate',preq) if KIND=='pd' else None
  q=urllib.request.Request(TARGET+'/generate',json.dumps(req).encode(),headers={'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(q,timeout=1800) as response:
    for rawline in response:
     line=rawline.decode().strip()
     if not line.startswith('data:'):continue
     raw=line[5:].strip()
     if raw=='[DONE]':break
     event=json.loads(raw);now=time.monotonic();events.append({'elapsed_s':now-started,'data':event});assert 'error' not in event,event
     n=len(event.get('output_ids',[]))
     if n>prev:
      if first is None:first,first_count=now,n
      elif last is not None:gaps.append((now-last)/(n-prev))
      last=now;prev=n
  except urllib.error.HTTPError as e:
   body=e.read().decode();append('http-errors.jsonl',{'label':label,'status':e.code,'body':body,'unix_s':time.time()});raise RuntimeError((e.code,body)) from e
  prefill=pf.result() if pf else None
 finally:pool.shutdown(wait=True)
 ended=time.monotonic();save(label+'.json',{'request':req,'prefill_request':preq if KIND=='pd' else None,'prefill_response':prefill,'events':events})
 assert events and prev==nout,(label,events[-1] if events else [])
 final=events[-1]['data']
 if prefill:assert len(prefill['output_ids'])==1 and prefill['output_ids'][0]==final['output_ids'][0],(label,prefill,final)
 row={'label':label,'adapter':adapter,'input_tokens':len(ids),'output_tokens':nout,'output_ids':final['output_ids'],'e2e_s':ended-started,'ttft_s':first-started,'tpot_s':(last-first)/(nout-first_count) if nout>first_count else None,'first_event_tokens':first_count,'started_monotonic':started,'ended_monotonic':ended,'meta_info':final['meta_info']}
 append('requests.jsonl',row);return row
def batch(nin,nout,adapters,label,logprobs=False):
 control('/flush_cache?timeout=60',{});barrier=threading.Barrier(len(adapters));started=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=len(adapters)) as pool:rows=list(pool.map(lambda pair:stream(prompt(nin,pair[0]),nout,pair[1],label+f'-i{pair[0]}',barrier,logprobs),enumerate(adapters)))
 ended=time.monotonic();group={'label':label,'input_tokens':nin,'output_tokens':nout,'concurrency':len(adapters),'elapsed_s':ended-started,'output_tokens_s':len(adapters)*nout/(ended-started),'started_monotonic':started,'ended_monotonic':ended}
 append('groups.jsonl',group);print(json.dumps(group),flush=True);return rows,group
def monitor():
 while not STOP.is_set():
  for role,url in URLS.items():
   try:
    t=time.monotonic();raw=call(url,'/metrics',timeout=10)
    with gzip.open(OUT/'raw-metrics.jsonl.gz','at') as f:f.write(json.dumps({'role':role,'started_monotonic':t,'ended_monotonic':time.monotonic(),'raw':raw})+'\n')
   except Exception as e:append('monitor-errors.jsonl',{'role':role,'error':repr(e),'unix_s':time.time()})
  STOP.wait(.5)
def main():
 global MON
 for attempt in range(450):
  try:
   infos={r:call(u,'/get_server_info',timeout=5) for r,u in URLS.items()}
   for u in URLS.values():call(u,'/health',timeout=5)
   if KIND=='pd':
    for role in ['prefill','decode']:
     logs=subprocess.check_output(['docker','logs',f'qwen38-fp8-mtp-{role}-r228'],stderr=subprocess.STDOUT,text=True)
     assert logs.rfind('End of disaggregation warmup')>logs.rfind('Start of pd disaggregation warmup')>=0,'warmup incomplete'
   break
  except Exception:time.sleep(2)
 else:raise TimeoutError('startup or PD warmup not ready')
 save('server-info.json',infos);save('placement.json',{'inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),'unix_s':time.time()})
 models={r:call(u,'/v1/models') for r,u in URLS.items()};save('models-before.json',models)
 for role,u in URLS.items():
  if not any(x.get('id')=='main' for x in models[role].get('data',[])):
   response=call(u,'/load_lora_adapter',{'lora_name':'main','lora_path':AD});append('controls.jsonl',{'role':role,'path':'/load_lora_adapter','response':response});assert response.get('success'),response
 MON=threading.Thread(target=monitor);MON.start()
 if PHASE=='smoke':
  batch(128,16,[None],'smoke-base',True);batch(128,16,['main'],'smoke-lora',True)
 elif PHASE=='speed':
  for nin,nout,count in [(128,128,1),(128,128,32),(128,128,64),(2048,256,32)]:
   batch(nin,nout,['main']*count,f'warm-in{nin}-c{count}')
   for repeat in range(3):batch(nin,nout,['main']*count,f'speed-in{nin}-c{count}-r{repeat}')
 elif PHASE=='capacity':
  for nin in [32768,131072,261632]:batch(nin,128,['main'],f'capacity-in{nin}-c1')
  batch(32768,1024,['main']*32,'capacity-in32768-c32')
 else:raise ValueError(PHASE)
 save('complete.json',{'passed':True,'scope':PHASE,'kind':KIND,'unix_s':time.time(),'note':'Native greedy generation, not merged-model precision acceptance or long-context task quality.'})
if __name__=='__main__':
 try:main()
 except BaseException:
  save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()});raise
 finally:
  STOP.set()
  if MON:MON.join(30)
