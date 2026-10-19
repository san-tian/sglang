"""Rebuild R55 lifecycle checks from every retained API response and stream event."""
import collections,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid(body,n):
    ids=body['output_ids'];lp=body['meta_info']['output_token_logprobs']
    assert len(ids)==len(lp)==n and [x[1] for x in lp]==ids
    assert all(math.isfinite(x[0]) for x in lp)
    assert body['meta_info']['completion_tokens']==n
    assert body['meta_info']['finish_reason']=={'type':'length','length':n}
    return [(x[0],x[1]) for x in lp]
def compare(body,ref,n):
    assert body['output_ids']==ref['output_ids'] and valid(body,n)==valid(ref,n)

def main():
    p=R/'active-lifecycle-r56';output=R/'active-lifecycle-raw-audit-r71.json';assert not output.exists()
    hashes={x.name:sha(x) for x in p.iterdir() if x.is_file()};assert len(hashes)==8
    api=lines(p/'api.jsonl');cfg=read(p/'config.json');summary=read(p/'summary.json')
    info=read(p/'server-info.json');server=info.get('server_args',info)
    assert server['tp_size']==server['attn_dp_size']==1 and server['enable_lora']
    assert server['lora_merged_bf16_cache_policy']=='active' and server['lora_merged_bf16_cache_size_mb']==36864
    assert server['max_total_num_tokens']==65536 and server['max_mamba_cache_size']==256 and server['mem_fraction_static']==.52
    assert cfg['cycles']==cfg['bad_loads']==100 and cfg['aliases']==12 and cfg['lru_rounds']==9
    assert cfg['aliases']>server['max_loaded_loras']
    adapters=cfg['adapters'];assert len(adapters)==3 and len({x['path'] for x in adapters})==3
    prefix=cfg['alias_prefix'];alias=prefix+'reuse';cursor=0;refs={};comparisons=0;scores=0
    def take(path,status=200):
        nonlocal cursor
        row=api[cursor];cursor+=1
        assert row['path']==path and row['status']==status,(cursor,path,status,row['path'],row['status'])
        assert math.isfinite(row['elapsed_s']) and row['elapsed_s']>=0
        return row
    def load(name,path):assert take('/load_lora_adapter')['request']=={'lora_name':name,'lora_path':path}
    def unload(name):
        row=take('/unload_lora_adapter');assert row['request']=={'lora_name':name};return row
    def generate(name,n=16,status=200):
        row=take('/generate',status);req=row['request']
        assert req=={'input_ids':cfg['input_ids'],'lora_path':name,'sampling_params':{'temperature':0,'max_new_tokens':n,'ignore_eos':True},'return_logprob':True,'logprob_start_len':-1}
        if status==200:
            valid(row['body'],n)
            assert row['body']['meta_info']['prompt_tokens']==len(cfg['input_ids'])
        return row['body']
    def check(body,ref,n):
        nonlocal comparisons,scores
        compare(body,ref,n);comparisons+=1;scores+=n
    assert take('/get_server_info')['body']==info
    for adapter in adapters:
        load(alias,adapter['path']);assert take('/flush_cache')['request']=={}
        refs[adapter['path']]=generate(alias);unload(alias)
    assert read(p/'baseline.json')==refs
    for i in range(100):
        path=adapters[i%3]['path'];load(alias,path);check(generate(alias),refs[path],16);unload(alias);generate(alias,status=400)
    load(alias,adapters[0]['path'])
    for _ in range(100):
        row=take('/load_lora_adapter',400)
        assert row['request']=={'lora_name':prefix+'bad','lora_path':cfg['missing_adapter']}
        check(generate(alias),refs[adapters[0]['path']],16)
    assert take('/flush_cache')['request']=={}
    inflight_ref=generate(alias,256);assert inflight_ref==read(p/'inflight-reference.json')
    assert take('/flush_cache')['request']=={}
    admitted=0
    while api[cursor]['path']=='/generate' and api[cursor]['status']==200:
        body=generate(alias,1);ref=refs[adapters[0]['path']]
        assert body['output_ids']==ref['output_ids'][:1] and valid(body,1)==valid(ref,16)[:1]
        admitted+=1
    generate(alias,1,status=400);drain=unload(alias)
    events=read(p/'inflight-events.json');assert events==lines(p/'inflight-events.jsonl')
    assert len(events[0]['output_ids'])==1
    previous=0
    for e in events:
        n=len(e['output_ids']);assert previous<=n<=256
        assert e['output_ids']==inflight_ref['output_ids'][:n]
        lp=e['meta_info']['output_token_logprobs'];assert len(lp)==n
        assert [(x[0],x[1]) for x in lp]==valid(inflight_ref,256)[:n]
        previous=n
    check(events[-1],inflight_ref,256)
    for i in range(12):load(prefix+f'lru_{i}',adapters[i%3]['path'])
    for _ in range(9):
        for i in range(12):check(generate(prefix+f'lru_{i}'),refs[adapters[i%3]['path']],16)
    for i in range(12):
        name=prefix+f'lru_{i}';unload(name);generate(name,status=400)
    take('/health');assert cursor==len(api)
    timing=summary['timing'];assert timing['first_token']<timing['unload_started']<timing['stream_done']<=timing['unload_done']
    assert math.isclose(timing['unload_done']-timing['unload_started'],drain['elapsed_s'],abs_tol=.01)
    for k,v in {'passed':True,'serial_cycles':100,'invalid_loads':100,'lru_requests':108,'lru_aliases':12,'inflight_output_tokens':256,'new_requests_rejected_while_draining':True,'max_repeat_logprob_abs_nat':0}.items():assert summary[k]==v
    assert comparisons==309 and scores==5184
    assert hashes=={n:sha(p/n) for n in hashes}
    counts=collections.Counter((x['path'],x['status']) for x in api)
    result={'passed':True,'api_records_reparsed':len(api),'api_counts':[{'path':k[0],'status':k[1],'count':v} for k,v in sorted(counts.items())],
            'reference_comparisons':comparisons,'exact_logprob_scores_compared':scores,'max_abs_nat':0,'natural_output_ids_equal':True,
            'race_requests_admitted_before_draining':admitted,'race_admitted_tokens_checked_against_reference':admitted,
            'stream_events_reparsed':len(events),'stream_output_tokens':256,'unload_elapsed_s':drain['elapsed_s'],
            'generation_after_unload_started_s':timing['stream_done']-timing['unload_started'],
            'raw_files_sha256':hashes,'audit_source_sha256':sha(Path(__file__)),
            'scope':'R55 active cache TP1, 100 same-alias replacements, 100 failed loads, 108 LRU recoveries, 256-token in-flight unload. Every API response replayed in operation order; natural IDs and original unforced API logprobs checked against same-service references. Drain timing is the retained client monotonic timing, cross-checked with API unload elapsed; raw API rows do not independently timestamp rejection. This is correctness evidence, not independent D/M precision, speedup or multi-GPU acceptance.'}
    output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['passed','api_records_reparsed','reference_comparisons','exact_logprob_scores_compared','stream_output_tokens','unload_elapsed_s']}))
if __name__=='__main__':main()
