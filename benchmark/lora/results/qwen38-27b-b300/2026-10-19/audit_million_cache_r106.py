"""Rehash six-arm original million-context SSE, recompute all statistics and controls."""
import hashlib,json,math
from pathlib import Path
from perf_million_cache_r106 import metrics,validate_enable_lora
R=Path(__file__).resolve().parent
source=json.loads((R/'source-milli-cache-r106.json').read_text())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_lora_role(cfg, side, model, launch):
    assert side in ('nocache','cached','m','base'), side
    enabled=side in ('nocache','cached')
    validate_enable_lora(cfg,enabled)
    expected='/work/qwen38-mol-development-20261019/merged/real_r16' if side=='m' else '/work/models/Qwen3.8-27B'
    assert model['model_path'].rstrip('/')==expected
    assert cfg['model_path'].rstrip('/')==expected
    assert ('--enable-lora' in launch)==enabled
    assert launch[launch.index('--model-path')+1].rstrip('/')==expected
    assert 'sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7' in launch

for name,h in source['source_files'].items():assert sha(R/name)==h,name
records={};configs={};hashes={};points=[];intervals=[]
for i,side in enumerate(['nocache','cached','m','base','cached','nocache']):
    d=R/f'perf-million-{i}-{side}-r97';cfg=read(d/'get_server_info.json');configs[str(i)]=cfg
    assert cfg['lora_merged_bf16_cache_size_mb']==(36864 if side=='cached' else 0)
    assert cfg['mem_fraction_static']==.8 and cfg['max_total_num_tokens']==1001024 and cfg['max_mamba_cache_size']==16
    assert cfg['tp_size']==1 and cfg['context_length']==1000000 and cfg['max_running_requests']==1 and cfg['chunked_prefill_size']==8192
    assert cfg['enable_deterministic_inference'] and cfg['disable_cuda_graph'] and cfg['dtype']=='bfloat16'
    assert json.loads(cfg['json_model_override_args'])==source['rope_override']
    assert (d/'corpus-sha256.txt').read_text().strip()==source['corpus_sha256']
    model=read(d/'model_info.json');expected='/work/qwen38-mol-development-20261019/merged/real_r16' if side=='m' else '/work/models/Qwen3.8-27B'
    assert model['model_path'].rstrip('/')==expected
    obs=read(R/f'qwen38-million-{i}-{side}-r97-actual-config.json');assert obs['image']==source['runtime_image']
    launch=read(R/f'launch-qwen38-million-{i}-{side}-r97.json')
    validate_lora_role(cfg,side,model,launch)
    rows=[json.loads(v) for v in (d/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==6 and {(v['round'],v['cache']) for v in rows}=={(n,c) for n in range(3) for c in ['cold','hot']}
    records[i]={};hashes[d.name]={};recomputed=[]
    for row in rows:
        label=f"r{row['round']}-{row['cache']}";p=d/(label+'.json');raw=read(p);assert raw['done'] and row['raw_sha256']==sha(p)
        measured=metrics(raw);assert all(measured[k]==row[k] for k in measured)
        assert raw['finished_unix_s']>=raw['started_unix_s'];intervals.append((raw['started_unix_s'],raw['finished_unix_s']))
        assert measured['cached_tokens']==0 if row['cache']=='cold' else 0<measured['cached_tokens']<999744
        req=raw['request'];assert req.get('lora_path')==('million-A' if side in ['nocache','cached'] else None)
        normalized={k:v for k,v in req.items() if k!='lora_path'}
        records[i][label]={'input_sha256':hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest(),'output_ids':measured['output_ids']}
        recomputed.append({**measured,'cache':row['cache']});hashes[d.name][p.name]=sha(p)
    warms=read(d/'warmup.json');assert len(warms)==2
    for n in range(2):
        p=d/f'warm-{n}.json';raw=read(p);assert raw['done'];measured=metrics(raw)
        assert measured['cached_tokens']==0 and all(measured[k]==warms[n][k] for k in measured)
        hashes[d.name][p.name]=sha(p)
    summ=read(d/'summary.json');assert summ['requests']==6 and summ['warmups']==2 and len(summ['points'])==2
    for cache in ['cold','hot']:
        rr=[v for v in recomputed if v['cache']==cache];point=next(v for v in summ['points'] if v['cache']==cache);assert point['requests']==3
        for k in ['ttft_s','tpot_s','e2e_s']:
            v=sum(x[k] for x in rr)/3;assert math.isfinite(v) and abs(v-point[k])<1e-8
        assert abs(sum(256/x['e2e_s'] for x in rr)/3-point['throughput_tokens_s_mean'])<1e-8
        points.append({'run':i,'side':side,**point,'cached_tokens_min_max':[min(x['cached_tokens'] for x in rr),max(x['cached_tokens'] for x in rr)]})
for label in records[2]:
    assert all(records[i][label]['input_sha256']==records[2][label]['input_sha256'] for i in records)
    assert all(records[i][label]['output_ids']==records[2][label]['output_ids'] for i in [0,1,4,5]),label
common=['tp_size','context_length','dtype','mem_fraction_static','max_total_num_tokens','max_mamba_cache_size','max_running_requests','chunked_prefill_size','json_model_override_args','attention_backend','linear_attn_prefill_backend','bf16_gemm_backend','enable_deterministic_inference','disable_cuda_graph']
for cfg in configs.values():assert all(cfg[k]==configs['0'][k] for k in common)
obs=R/'million-cache-gpu-observations-r97.jsonl';samples=0;times=[]
with obs.open() as f:
    for line in f:
        row=json.loads(line);assert 'error' not in row and not row['unexpected_active_gpus'] and len(row['rows'])==8
        assert all(int(v[1])<2000 and int(v[2])==0 for v in row['rows'] if int(v[0])!=2);samples+=1;times.append(row['unix_s'])
assert samples>1 and times[0]<=min(x[0] for x in intervals) and times[-1]>=max(x[1] for x in intervals)
assert all(0<b-a<=15 for a,b in zip(times,times[1:])), 'GPU observation gap over15s'
result={'passed':True,'native_requests':36,'warmups':12,'points':points,'request_and_natural_output_equivalence':True,'original_file_sha256':hashes,'one_second_gpu_samples':samples,'gpu_observations_sha256':sha(obs),'source_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(R/'source-milli-cache-r106.json'),'original_source_manifest_sha256':sha(R/'source-milli-cache-r97.json'),'scope':source['scope']}
out=R/'million-cache-comparison-r97.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'native_requests':36,'points':points}))
