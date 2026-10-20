"""Auditable native PD speed, KV residency, and distinct scaled-LoRA capacity."""
import concurrent.futures as cf
import copy,gzip,hashlib,json,math,re,threading,time,traceback,urllib.request,uuid
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
OUT=R/'measure-1p1d-r160'
P='http://127.0.0.1:31800'; D='http://127.0.0.1:31801'
AD='/work/qwen38-mol-experiment/adapters/real_r16'
def save(name,obj):
 path=OUT/name
 with path.open('x') as f:json.dump(obj,f,ensure_ascii=False)
def append(name,obj):
 with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n')
def call(url,path,data=None,timeout=1200):
 req=urllib.request.Request(url+path,None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as f:raw=f.read().decode()
 try:return json.loads(raw)
 except ValueError:return raw
def control(path,data=None):
 started=time.time()
 with cf.ThreadPoolExecutor(max_workers=2) as pool:
  values=list(pool.map(lambda u:call(u,path,data),[P,D]))
 append('controls.jsonl',{'unix_s':started,'path':path,'request':data,'responses':values})
 if path=='/load_lora_adapter':assert all(v.get('success') for v in values),values
 return values
def prompt(n,i=0):
 ids=(FILLER*((n+len(FILLER)-1)//len(FILLER)))[:n]
 ids[:8]=[1000+64*i+j for j in range(8)]
 return ids
def stream(ids,nout,adapter,label,barrier=None,logprobs=False):
 req={'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':nout,'ignore_eos':True},'bootstrap_host':'127.0.0.1','bootstrap_port':31810,'bootstrap_room':uuid.uuid4().int&((1<<63)-1)}
 if adapter is not None:req['lora_path']=adapter
 if logprobs:req['return_logprob']=True
 preq=copy.deepcopy(req);preq['stream']=False
 if barrier:barrier.wait()
 started=time.monotonic();first=last=None;first_count=prev=0;events=[];gaps=[]
 with cf.ThreadPoolExecutor(max_workers=1) as pool:
  pf=pool.submit(call,P,'/generate',preq)
  q=urllib.request.Request(D+'/generate',json.dumps(req).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(q,timeout=1800) as response:
   for rawline in response:
    line=rawline.decode().strip()
    if not line.startswith('data:'):continue
    raw=line[5:].strip()
    if raw=='[DONE]':break
    event=json.loads(raw);now=time.monotonic();events.append({'elapsed_s':now-started,'data':event})
    assert 'error' not in event,event
    n=len(event.get('output_ids',[]))
    if n>prev:
     if first is None:first,first_count=now,n
     elif last is not None:gaps.append((now-last)/(n-prev))
     last=now;prev=n
  prefill=pf.result()
 ended=time.monotonic();assert events and prev==nout,(label,events[-1] if events else [])
 final=events[-1]['data'];assert len(prefill['output_ids'])==1 and prefill['output_ids'][0]==final['output_ids'][0]
 save(label+'.json',{'decode_request':req,'prefill_request':preq,'prefill_response':prefill,'decode_events':events})
 row={'label':label,'adapter':adapter,'input_tokens':len(ids),'output_tokens':nout,'output_ids':final['output_ids'],'e2e_s':ended-started,'ttft_s':first-started,'tpot_s':(last-first)/(nout-first_count),'event_normalized_gaps_s':gaps,'first_event_tokens':first_count,'cached_tokens_prefill':prefill['meta_info'].get('cached_tokens',0),'cached_tokens_decode':final['meta_info'].get('cached_tokens',0),'started_monotonic':started,'ended_monotonic':ended}
 if logprobs:row['output_token_logprobs']=final['meta_info'].get('output_token_logprobs')
 append('requests.jsonl',row)
 return row
def batch(nin,nout,adapters,label,logprobs=False):
 control('/flush_cache',{})
 barrier=threading.Barrier(len(adapters));started=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=len(adapters)) as pool:
  rows=list(pool.map(lambda pair:stream(prompt(nin,pair[0]),nout,pair[1],label+f'-i{pair[0]}',barrier,logprobs),enumerate(adapters)))
 ended=time.monotonic()
 group={'label':label,'input_tokens':nin,'output_tokens':nout,'concurrency':len(adapters),'distinct_adapters':len(set(adapters)),'started_monotonic':started,'ended_monotonic':ended,'elapsed_s':ended-started,'output_tokens_s':len(adapters)*nout/(ended-started)}
 append('groups.jsonl',group);print(json.dumps(group),flush=True)
 return rows,group
def monitor():
 while not STOP.is_set():
  for role,url in [('prefill',P),('decode',D)]:
   started=time.monotonic()
   try:
    raw=call(url,'/metrics',timeout=10)
    row={'role':role,'started_monotonic':started,'ended_monotonic':time.monotonic(),'raw':raw}
    with gzip.open(OUT/'raw-metrics.jsonl.gz','at') as f:f.write(json.dumps(row)+'\n')
   except Exception as e:append('monitor-errors.jsonl',{'role':role,'error':repr(e),'unix_s':time.time()})
  STOP.wait(.5)
def make_scaled():
 src=Path('/opt/dlami/nvme/qwen38-mol-experiment/adapters/real_r16')
 dest=R/'adapters-r160';dest.mkdir(exist_ok=False)
 config=json.loads((src/'adapter_config.json').read_text())
 result=[]
 for i in range(32):
  folder=dest/f'scaled-{i:02d}';folder.mkdir()
  cfg=dict(config);cfg['lora_alpha']=float(config['lora_alpha'])*(i+1)/32
  (folder/'adapter_config.json').write_text(json.dumps(cfg,indent=2))
  for f in src.iterdir():
   if f.name!='adapter_config.json':(folder/f.name).symlink_to(f)
  result.append({'name':f'slot-{i:02d}','path':str(folder).replace('/opt/dlami/nvme','/work',1),'alpha':cfg['lora_alpha'],'kind':'distinct synthetic alpha-scaled real checkpoint; common A/B source'})
 return result
def main():
 OUT.mkdir(exist_ok=False)
 for _ in range(900):
  try:
   infos={role:call(url,'/get_server_info',timeout=5) for role,url in [('prefill',P),('decode',D)]}
   for url in [P,D]:call(url,'/health',timeout=5)
   break
  except Exception:time.sleep(2)
 else:raise TimeoutError('1P1D startup')
 save('server-info.json',infos)
 import subprocess
 save('placement.json',{'gpu_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),'inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),'containers':json.loads(subprocess.check_output(['docker','inspect','qwen38-1p1d-prefill-r160','qwen38-1p1d-decode-r160'],text=True))})
 control('/load_lora_adapter',{'lora_name':'main','lora_path':AD})
 thread=threading.Thread(target=monitor);thread.start()
 try:
  batch(128,16,['main'],'smoke',True)
  for nin,nout in [(128,128),(2048,256)]:
   for count in [1,32,64]:
    batch(nin,nout,['main']*count,f'warm-in{nin}-c{count}')
    for repeat in range(3):batch(nin,nout,['main']*count,f'speed-in{nin}-c{count}-r{repeat}')
  for nin,count in [(32768,1),(8192,8),(8192,32)]:
   batch(nin,512,['main']*count,f'capacity-in{nin}-c{count}')
  control('/unload_lora_adapter',{'lora_name':'main'})
  scaled=make_scaled();save('scaled-adapters.json',scaled)
  for item in scaled:control('/load_lora_adapter',{'lora_name':item['name'],'lora_path':item['path']})
  save('loaded-32-server-info.json',{role:call(url,'/get_server_info') for role,url in [('prefill',P),('decode',D)]})
  # Same prompt and natural logprobs separately vs mixed, including adapters outside the GPU pool.
  refs=[]
  for i in [0,7,15,23,31]:
   control('/flush_cache',{})
   refs.append(stream(prompt(128,i),16,scaled[i]['name'],f'slot-reference-{i}',logprobs=True))
  save('slot-reference-outputs.json',refs)
  for count in [1,4,8,16,32]:
   for repeat in range(2):batch(128,32,[x['name'] for x in scaled[:count]],f'slots-n{count}-r{repeat}',True)
  # Explicit base competes with 16 adapter identities for 16 GPU slots; scheduler must finish safely.
  batch(128,32,[None]+[x['name'] for x in scaled[:16]],'slots-base-plus16',True)
  save('complete.json',{'passed':True,'full_goal_complete':False,'completed_unix_s':time.time(),'scope':'R55 BF16 deterministic eager 1P1D; synthetic alpha-scaled adapters for capacity, not trained-model quality; raw residency requires independent metric audit; no independent merged precision acceptance yet'})
 finally:
  STOP.set();thread.join(30)
if __name__=='__main__':
 STOP=threading.Event()
 corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text())
 FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
 try:main()
 except BaseException:
  if OUT.exists():save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()})
  raise
