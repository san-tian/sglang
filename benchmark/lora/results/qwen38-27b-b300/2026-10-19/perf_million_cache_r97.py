"""Single endpoint, natural million-total-context SSE control; no profiler/scorer."""
import argparse,hashlib,json,math,time,urllib.request
from pathlib import Path

def call(url,path,data=None):
    body=None if data is None else json.dumps(data).encode()
    req=urllib.request.Request(url+path,body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=7200) as f:
        raw=f.read();return json.loads(raw) if raw else None

def metrics(raw,nin=999744,nout=256):
    req=raw['request'];assert req['stream'] is True and 'custom_logit_processor' not in req
    assert len(req['input_ids'])==nin and req['sampling_params']=={'temperature':0,'ignore_eos':True,'max_new_tokens':nout}
    prev=[];increasing=[];last_t=-1
    for event in raw['events']:
        t=event['elapsed_s'];assert math.isfinite(t) and t>=last_t>=-1;last_t=t
        ids=event['data'].get('output_ids',[]);assert ids[:len(prev)]==prev and len(ids)>=len(prev)
        if len(ids)>len(prev):increasing.append(event)
        prev=ids
    assert increasing and len(increasing[0]['data']['output_ids'])==1
    final=raw['events'][-1]['data'];meta=final['meta_info'];assert len(final['output_ids'])==nout
    assert meta['prompt_tokens']==nin and meta['completion_tokens']==nout and meta['finish_reason']=={'type':'length','length':nout}
    first=increasing[0]['elapsed_s'];last=increasing[-1]['elapsed_s'];elapsed=raw['http_elapsed_s']
    assert math.isfinite(elapsed) and elapsed>=last_t and first>0 and last>first
    return {'ttft_s':first,'tpot_s':(last-first)/(nout-1),'e2e_s':elapsed,'output_ids':final['output_ids'],'cached_tokens':meta.get('cached_tokens',0)}

def stream(url,ids,label,out,adapter=False):
    req={'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':256}}
    if adapter:req['lora_path']='million-A'
    events=[];started_unix_s=time.time();start=time.monotonic();done=False
    request=urllib.request.Request(url+'/generate',json.dumps(req).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=7200) as f:
            for line in f:
                text=line.decode().strip()
                if not text.startswith('data:'):continue
                payload=text[5:].strip()
                if payload=='[DONE]':done=True;break
                events.append({'elapsed_s':time.monotonic()-start,'data':json.loads(payload)})
    except BaseException as error:
        (out/(label+'-failed.json')).write_text(json.dumps({'request':req,'events':events,'http_elapsed_s':time.monotonic()-start,'error':repr(error)}))
        raise
    raw={'request':req,'events':events,'http_elapsed_s':time.monotonic()-start,'done':done,'started_unix_s':started_unix_s,'finished_unix_s':time.time()}
    path=out/(label+'.json');path.write_text(json.dumps(raw));assert done,'Incomplete SSE stream'
    return {**metrics(raw),'label':label,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter-path');p.add_argument('--corpus',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);data=json.loads(a.corpus.read_text());assert len(data)==1
    ids=data[0]['input_ids'];assert len(ids)==999744
    (a.output/'corpus-sha256.txt').write_text(hashlib.sha256(a.corpus.read_bytes()).hexdigest()+'\n')
    for end in ['get_server_info','model_info']:(a.output/(end+'.json')).write_text(json.dumps(call(a.url,'/'+end),indent=2))
    cfg=json.loads((a.output/'get_server_info.json').read_text());assert cfg['context_length']==1000000
    assert cfg['max_total_num_tokens']==1001024 and cfg['max_mamba_cache_size']==16
    assert cfg['tp_size']==1 and cfg['mem_fraction_static']==.8
    assert cfg['enable_lora']==bool(a.adapter_path)
    loaded=False
    try:
        if a.adapter_path:call(a.url,'/load_lora_adapter',{'lora_name':'million-A','lora_path':a.adapter_path});loaded=True
        warms=[]
        for n in range(2):
            call(a.url,'/flush_cache',{});warms.append(stream(a.url,ids,f'warm-{n}',a.output,loaded))
        (a.output/'warmup.json').write_text(json.dumps(warms,indent=2))
        rows=[]
        with (a.output/'requests.jsonl').open('x') as f:
            for repeat in range(3):
                for cache in ['cold','hot']:
                    if cache=='cold':call(a.url,'/flush_cache',{})
                    row=stream(a.url,ids,f'r{repeat}-{cache}',a.output,loaded)
                    assert row['cached_tokens']==0 if cache=='cold' else 0<row['cached_tokens']<len(ids)
                    row.update(round=repeat,cache=cache);rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                    print(json.dumps({k:v for k,v in row.items() if k!='output_ids'}),flush=True)
        points=[]
        for cache in ['cold','hot']:
            rr=[v for v in rows if v['cache']==cache]
            points.append({'cache':cache,'requests':3,**{k:sum(v[k] for v in rr)/3 for k in ['ttft_s','tpot_s','e2e_s']},'throughput_tokens_s_mean':sum(256/v['e2e_s'] for v in rr)/3})
        (a.output/'summary.json').write_text(json.dumps({'requests':6,'warmups':2,'points':points,'scope':'One fixed999744-token input/256 natural outputs, c1,3cold/hot repeats. KV cold/hot is separate from projection warmup; two cold KV warmups precede measurement. No full-prefill logprob or task-quality acceptance.'},indent=2))
    finally:
        if loaded:call(a.url,'/unload_lora_adapter',{'lora_name':'million-A'})
if __name__=='__main__':main()
