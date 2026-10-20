"""Native SGLang 4P4D baseline; independent P/D dispatch, complete raw evidence."""
import concurrent.futures as cf
import copy,gzip,hashlib,json,math,subprocess,sys,threading,time,traceback,urllib.request,urllib.error,uuid
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
PHASE=sys.argv[1];OUT=E/PHASE;OUT.mkdir(exist_ok=False)
(OUT/'measurement-source.py').write_bytes(Path(__file__).read_bytes())
TOPO=json.loads((E/'topology.json').read_text());WORKERS=TOPO['workers']
PS=[w for w in WORKERS if w['role']=='prefill'];DS=[w for w in WORKERS if w['role']=='decode']
AD='/work/qwen38-mol-experiment/adapters/real_r16';STOP=threading.Event();LOCK=threading.Lock()
PPOOL=cf.ThreadPoolExecutor(max_workers=256)
corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text())
FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
CASES=json.loads((R/'precision-1p1d-reference-r168'/'contract.json').read_text())['corpus']
def save(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,ensure_ascii=False)
def append(n,v):
 with LOCK:
  with (OUT/n).open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
def call(url,path,data=None,timeout=1200):
 q=urllib.request.Request(url+path,None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(q,timeout=timeout) as f:raw=f.read().decode()
 except urllib.error.HTTPError as e:
  body=e.read().decode();append('http-errors.jsonl',{'url':url+path,'status':e.code,'body':body,'unix_s':time.time()});raise
 try:return json.loads(raw)
 except ValueError:return raw
def control(path,data=None):
 with cf.ThreadPoolExecutor(max_workers=8) as pool:
  values=list(pool.map(lambda w:{'worker':w['name'],'response':call(w['url'],path,data)},WORKERS))
 append('controls.jsonl',{'unix_s':time.time(),'path':path,'request':data,'responses':values})
 if path in ['/load_lora_adapter','/unload_lora_adapter']:assert all(x['response'].get('success') for x in values),values
 return values
def prompt(n,i):
 ids=(FILLER*((n+len(FILLER)-1)//len(FILLER)))[:n];ids[:8]=[1000+64*i+j for j in range(8)];return ids
def stream(ids,nout,adapter,label,pidx,didx,barrier=None,logprobs=False,ignore_eos=True):
 p,d=PS[pidx],DS[didx]
 req={'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':nout,'ignore_eos':ignore_eos},
      'bootstrap_host':'127.0.0.1','bootstrap_port':p['bootstrap_port'],'bootstrap_room':uuid.uuid4().int&((1<<63)-1)}
 if adapter is not None:req['lora_path']=adapter
 if logprobs:req.update(return_logprob=True,logprob_start_len=0,top_logprobs_num=1)
 preq=copy.deepcopy(req);preq['stream']=False
 if barrier:barrier.wait(timeout=120)
 started=time.monotonic();first=last=None;first_count=prev=0;events=[];gaps=[];prefill=None;error=None
 pf=PPOOL.submit(call,p['url'],'/generate',preq)
 try:
  q=urllib.request.Request(d['url']+'/generate',json.dumps(req).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(q,timeout=1800) as response:
   for line in response:
    line=line.decode().strip()
    if not line.startswith('data:'):continue
    raw=line[5:].strip()
    if raw=='[DONE]':break
    ev=json.loads(raw);now=time.monotonic();events.append({'elapsed_s':now-started,'data':ev});assert 'error' not in ev,ev
    n=len(ev.get('output_ids',[]))
    if n>prev:
     if first is None:first,first_count=now,n
     elif last is not None:gaps.append({'gap_s':now-last,'new_tokens':n-prev})
     last=now;prev=n
  prefill=pf.result(timeout=1800)
  assert events and prev>0 and (prev==nout or not ignore_eos),(label,events[-1] if events else [])
  assert len(prefill['output_ids'])==1 and prefill['output_ids'][0]==events[-1]['data']['output_ids'][0],(label,prefill)
 except BaseException:
  error=traceback.format_exc();raise
 finally:
  ended=time.monotonic()
  with gzip.open(OUT/(label+'.json.gz'),'xt') as f:json.dump({'request':req,'prefill_request':preq,'prefill_response':prefill,'events':events,'error':error,'p':p,'d':d},f,ensure_ascii=False)
  if error:append('request-errors.jsonl',{'label':label,'error':error,'p':pidx,'d':didx})
 final=events[-1]['data'];row={'label':label,'adapter':adapter,'p':pidx,'d':didx,'input_tokens':len(ids),'output_tokens':prev,
  'output_ids':final['output_ids'],'e2e_s':ended-started,'ttft_s':first-started,'tpot_s':(last-first)/(prev-first_count) if prev>first_count else None,
  'event_gaps':gaps,'first_event_tokens':first_count,'started_monotonic':started,'ended_monotonic':ended,'meta_info':final['meta_info'],'prefill_meta':prefill['meta_info']}
 append('requests.jsonl',row);return row
def batch(nin,nout,adapters,label,offset=0,natural=False):
 control('/flush_cache?timeout=60',{});barrier=threading.Barrier(len(adapters));started=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=len(adapters)) as pool:
  fs=[pool.submit(stream,CASES[i%len(CASES)]['input_ids'] if natural else prompt(nin,i),nout,a,label+f'-i{i}',(i+offset)%4,(i%4+i//4+offset)%4,barrier,False,not natural) for i,a in enumerate(adapters)]
  rows=[f.result() for f in fs]
 ended=time.monotonic();g={'label':label,'input_tokens':nin if not natural else 'natural-short','max_output_tokens':nout,'concurrency':len(adapters),
  'elapsed_s':ended-started,'output_tokens':sum(x['output_tokens'] for x in rows),'input_tokens_total':sum(x['input_tokens'] for x in rows),
  'started_monotonic':started,'ended_monotonic':ended,'p_counts':[sum(x['p']==i for x in rows) for i in range(4)],'d_counts':[sum(x['d']==i for x in rows) for i in range(4)]}
 g['output_tokens_s']=g['output_tokens']/g['elapsed_s'];g['input_tokens_s']=g['input_tokens_total']/g['elapsed_s']
 for k in ['ttft_s','tpot_s','e2e_s']:
  vals=sorted(x[k] for x in rows if x[k] is not None)
  g[k]={'mean':sum(vals)/len(vals),'p50':vals[math.ceil(.5*len(vals))-1],'p95':vals[math.ceil(.95*len(vals))-1],'p99':vals[math.ceil(.99*len(vals))-1]} if vals else None
 append('groups.jsonl',g);print(json.dumps(g),flush=True);return rows,g
def monitor():
 with cf.ThreadPoolExecutor(max_workers=8) as pool:
  while not STOP.is_set():
   def snap(w):
    try:
     t=time.monotonic();raw=call(w['url'],'/metrics',timeout=10)
     return {'worker':w['name'],'role':w['role'],'index':w['index'],'started_monotonic':t,'ended_monotonic':time.monotonic(),'raw':raw}
    except Exception as e:return {'worker':w['name'],'error':repr(e),'unix_s':time.time()}
   with gzip.open(OUT/'raw-metrics.jsonl.gz','at') as f:
    for r in pool.map(snap,WORKERS):f.write(json.dumps(r)+'\n')
   try:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True)
    append('gpu-metrics.jsonl',{'monotonic':time.monotonic(),'raw':raw})
   except Exception as e:append('monitor-errors.jsonl',{'error':repr(e)})
   STOP.wait(.5)
def main():
 for attempt in range(600):
  try:
   infos={w['name']:call(w['url'],'/get_server_info',timeout=3) for w in WORKERS}
   for w in WORKERS:
    call(w['url'],'/health',timeout=3)
    log=subprocess.check_output(['docker','logs',w['name']],stderr=subprocess.STDOUT,text=True)
    assert log.rfind('End of disaggregation warmup')>log.rfind('Start of pd disaggregation warmup')>=0,w['name']
   break
  except Exception as e:
   if attempt%10==0:print('waiting readiness',repr(e),flush=True)
   time.sleep(2)
 else:raise TimeoutError('eight workers not ready')
 save('server-info.json',infos)
 for w in WORKERS:
  models=call(w['url'],'/v1/models')
  if not any(x.get('id')=='main' for x in models.get('data',[])):
   v=call(w['url'],'/load_lora_adapter',{'lora_name':'main','lora_path':AD});assert v.get('success'),v
 mon=threading.Thread(target=monitor);mon.start()
 try:
  if PHASE=='smoke':
   for adapter in [None,'main']:
    for p in range(4):
     for d in range(4):stream(CASES[0]['input_ids'],32,adapter,f'pair-{adapter}-p{p}-d{d}',p,d,logprobs=True)
   for i,c in enumerate(CASES):
    for d in range(4):stream(c['input_ids'],32,'main',f'natural-i{i}-d{d}',i%4,d,logprobs=True)
  elif PHASE=='speed':
   configs=[(128,128,c) for c in [1,32,64,128,256]]+[(2048,256,c) for c in [32,128,256]]+[(8192,1024,c) for c in [32,128]]+[(32768,1024,c) for c in [32,64]]
   save('contract.json',{'configs':configs,'warmups':1,'rounds':3,'dispatch':'P=(i+round)%4,D=(i%4+i//4+round)%4','flush_each_batch':True,'workload':'repeated filler; unique first8 IDs; greedy ignore_eos; native client overhead included','baseline_accuracy_accepted':False})
   for nin,nout,c in configs:
    batch(nin,nout,['main']*c,f'warm-in{nin}-out{nout}-c{c}')
    for r in range(3):batch(nin,nout,['main']*c,f'speed-in{nin}-out{nout}-c{c}-r{r}',r)
  elif PHASE=='capacity':
   for nin in [32768,131072,262016]:
    batch(nin,128,['main']*4,f'capacity-in{nin}-c4')
   for c in [32,64]:batch(131072,512,['main']*c,f'capacity-in131072-c{c}')
  elif PHASE=='natural':
   for c in [32,128]:
    batch(0,256,['main']*c,f'warm-natural-c{c}',natural=True)
    for r in range(3):batch(0,256,['main']*c,f'natural-c{c}-r{r}',r,natural=True)
  elif PHASE=='slots':
   scaled=[{'name':f'slot-{i:02d}','path':f'/work/qwen38-mol-multigpu-20261019/fp8-mtp-r228/adapters-r239/scaled-{i:02d}'} for i in range(32)]
   control('/unload_lora_adapter',{'lora_name':'main'})
   try:
    for s in scaled:control('/load_lora_adapter',{'lora_name':s['name'],'lora_path':s['path']})
    save('models-loaded32.json',{w['name']:call(w['url'],'/v1/models') for w in WORKERS})
    save('server-info-loaded32.json',{w['name']:call(w['url'],'/get_server_info') for w in WORKERS})
    # Each D sees all 32 identities, not just one quarter of them.
    mixed,_=batch(128,32,[s['name'] for s in scaled for _ in range(4)],'slots32-eachD-c128')
    # All 17 identities (base+16 adapters) target each D, exceeding its16 GPU budget.
    batch(128,32,[s for s in [None]+[x['name'] for x in scaled[:16]] for _ in range(4)],'base-plus16-eachD-c68')
    refs=[]
    for i in [0,15,31]:
     for d in range(4):
      k=(d-i)%4;idx=4*i+k
      ref=stream(prompt(128,idx),32,scaled[i]['name'],f'slot-reference-{i}-d{d}',k,d,logprobs=True)
      refs.append({'identity':i,'d':d,'mixed_label':mixed[idx]['label'],'same_output_ids':ref['output_ids']==mixed[idx]['output_ids'],'request':ref})
    save('slot-references.json',refs)
   finally:
    for s in scaled:
     try:control('/unload_lora_adapter',{'lora_name':s['name']})
     except Exception as e:append('cleanup-errors.jsonl',{'name':s['name'],'error':repr(e)})
    control('/load_lora_adapter',{'lora_name':'main','lora_path':AD})
  else:raise ValueError(PHASE)
  save('complete.json',{'passed':True,'phase':PHASE,'unix_s':time.time(),'scope':'4P4D TP1 baseline; no new independent FP8 merged precision acceptance'})
 finally:STOP.set();mon.join(30)
if __name__=='__main__':
 try:main()
 except BaseException:
  save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()});raise
 finally:PPOOL.shutdown(wait=True)
