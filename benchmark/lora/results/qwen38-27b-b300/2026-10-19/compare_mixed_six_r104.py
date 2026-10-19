"""Reconstruct complete, counterbalanced R84 policy results from original SSE.

Requires every cohort's independent R93 reconstruction and the full R105
recheck of original R84 evidence; retains the original failed audit marker. Produces an observation, not a default-policy or isolated-speed gate.
"""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path

POLICIES = ('nocache', 'lru', 'active', 'active', 'lru', 'nocache')
IMAGE = 'sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
COMMON = ('tp_size','attn_dp_size','context_length','dtype','max_total_num_tokens',
          'max_mamba_cache_size','mem_fraction_static','attention_backend','page_size',
          'chunked_prefill_size','max_running_requests','enable_deterministic_inference',
          'disable_cuda_graph','random_seed')

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in p.read_text().splitlines()]
def close(x,y):
    assert math.isfinite(x) and math.isfinite(y)
    assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8),(x,y)
def percentile(values,q):
    assert values and all(math.isfinite(x) for x in values)
    return sorted(values)[max(0,math.ceil(len(values)*q)-1)]

def compare(root):
    marker=root/'active-cache-recovery-finished-r84.json'
    finished=read(marker)
    assert finished['passed'] is False and 'audit_active_cache_suite_r84.py' in finished['error']
    suitefile=root/'active-cache-suite-recheck-r105.json';suite=read(suitefile)
    assert suite['original_failed_marker_sha256']==sha(marker)
    assert suite['source_sha256']==sha(root/'audit_active_cache_recheck_r105.py')
    assert suite['original_audit_source_sha256']==sha(root/'audit_active_cache_suite_r84.py')
    assert suite['passed'] and suite['mixed_native_requests_total']==1152
    assert suite['native_outputs_identical_across_six_mixed_runs']
    records={};configs=[];cohorts=[];raw_rows={};batch_groups={};hashes={}
    for index,policy in enumerate(POLICIES):
        folder=root/f'perf-active-mix-{index}-{policy}-r84'
        afile=root/f'mixed-cohort-audit-{index}-{policy}-r93.json';audit=read(afile)
        assert audit['passed'] and audit['run']==index and audit['policy']==policy
        assert audit['natural_requests']==192 and audit['warmup_batches']==2
        assert audit['actual_runtime_image']==IMAGE
        assert sha(folder/'summary.json')==audit['summary_sha256']
        config=read(folder/'server-info.json');config=config.get('server_args',config)
        assert config['max_total_num_tokens']==65536 and config['max_mamba_cache_size']==256
        assert config['mem_fraction_static']==.52 and config['tp_size']==config['attn_dp_size']==1
        assert config['lora_merged_bf16_cache_size_mb']==(0 if policy=='nocache' else 36864)
        assert config['lora_merged_bf16_cache_policy']==('lru' if policy=='nocache' else policy)
        configs.append(config)
        rows=lines(folder/'requests.jsonl');groups=lines(folder/'groups.jsonl')
        assert len(rows)==len({v['label'] for v in rows})==192
        assert len(groups)==len({(g['round'],g['cache']) for g in groups})==6
        assert {(g['round'],g['cache']) for g in groups}=={(n,c) for n in range(3) for c in ('cold','hot')}
        records[index]={}
        for row in rows:
            file=folder/(row['label']+'.json')
            assert sha(file)==audit['original_request_files_sha256'][file.name]
            raw=read(file);req=raw['request'];events=raw['events']
            assert len(req['input_ids'])==2048 and req['stream'] is True
            assert req['sampling_params']=={'temperature':0,'max_new_tokens':256,'ignore_eos':True}
            prev=[];clock=-1;first=last=None
            for e in events:
                ids=e['data']['output_ids'];t=e['elapsed_s']
                assert math.isfinite(t) and t>=clock and len(ids)>=len(prev) and ids[:len(prev)]==prev
                if len(ids)>len(prev):
                    if first is None:
                        assert len(ids)==1;first=t
                    last=t
                prev=ids;clock=t
            assert len(prev)==256 and prev==row['output_ids']
            meta=events[-1]['data']['meta_info']
            assert meta['prompt_tokens']==2048 and meta['completion_tokens']==256
            assert meta.get('cached_tokens')==row['cached_tokens']
            close(first,row['ttft_s']);close((last-first)/255,row['tpot_s'])
            assert math.isfinite(row['e2e_s']) and row['e2e_s']>=last
            assert events[-1]['data']['meta_info']['finish_reason']=={'type':'length','length':256}
            records[index][row['label']]={'request':req,'output_ids':prev}
        assert len(audit['original_request_files_sha256'])==192
        for g in groups:
            subset=[v for v in rows if (v['round'],v['cache'])==(g['round'],g['cache'])]
            assert len(subset)==32 and collections.Counter(v['identity'] for v in subset)=={'base':8,'real':8,'half':8,'gdn':8}
            assert g['elapsed_s']>=max(v['e2e_s'] for v in subset)
            close(8192/g['elapsed_s'],g['output_throughput_tok_s'])
        # R93's per-cohort percentiles/means must match independently again.
        for point in audit['points']:
            cache=point['cache'];subset=[v for v in rows if v['cache']==cache]
            selected=[g for g in groups if g['cache']==cache]
            close(statistics.mean(8192/g['elapsed_s'] for g in selected),point['throughput_tokens_s_mean'])
            for metric in ('ttft_s','tpot_s','e2e_s'):
                for name,q in [('p50',.5),('p95',.95),('p99',.99)]:
                    close(percentile([v[metric] for v in subset],q),point[metric][name])
        for file,h in audit['original_request_files_sha256'].items(): assert sha(folder/file)==h
        raw_rows[index]=rows;batch_groups[index]=groups
        hashes[str(index)]={'cohort_audit_sha256':sha(afile),'summary_sha256':audit['summary_sha256'],
                            'raw_requests_rehashed':192,'groups_sha256':sha(folder/'groups.jsonl')}
        cohorts.append({'run':index,'policy':policy,'points':audit['points']})
    assert all(records[i]==records[0] for i in range(6)), 'payload or natural output changed'
    assert all(all(c[k]==configs[0][k] for k in COMMON) for c in configs)
    pooled=[]
    for policy in ('nocache','lru','active'):
        indexes=[i for i,p in enumerate(POLICIES) if p==policy]
        for cache in ('cold','hot'):
            rows=[v for i in indexes for v in raw_rows[i] if v['cache']==cache]
            groups=[g for i in indexes for g in batch_groups[i] if g['cache']==cache]
            assert len(rows)==192 and len(groups)==6
            rates=[8192/g['elapsed_s'] for g in groups]
            hits=[v['cached_tokens'] for v in rows if v['cached_tokens'] is not None]
            point={'policy':policy,'cache':cache,'requests':192,'rounds':6,
                   'batch_throughput_tokens_s':rates,
                   'mean_batch_throughput_tokens_s':statistics.mean(rates),
                   'pooled_throughput_tokens_s':49152/sum(g['elapsed_s'] for g in groups),
                   'batch_rate_min_max':[min(rates),max(rates)],
                   'cached_tokens_min_max':[min(hits),max(hits)] if hits else None}
            for metric in ('ttft_s','tpot_s','e2e_s'):
                point[metric]={'mean':statistics.mean(v[metric] for v in rows),
                               **{name:percentile([v[metric] for v in rows],q) for name,q in [('p50',.5),('p95',.95),('p99',.99)]}}
            pooled.append(point)
    ratios=[]
    for cache in ('cold','hot'):
        means={v['policy']:v['mean_batch_throughput_tokens_s'] for v in pooled if v['cache']==cache}
        ratio={'cache':cache,'pooled_mean_active_over_LRU':means['active']/means['lru'],
               'pooled_mean_active_over_nocache':means['active']/means['nocache'],'order_observations':[]}
        for order,indexes in [('forward',{'nocache':0,'lru':1,'active':2}),('reverse',{'active':3,'lru':4,'nocache':5})]:
            controls={p:next(v for v in cohorts[i]['points'] if v['cache']==cache)['throughput_tokens_s_mean'] for p,i in indexes.items()}
            ratio['order_observations'].append({'order':order,'controls':controls,'active_over_LRU':controls['active']/controls['lru'],'active_over_nocache':controls['active']/controls['nocache']})
        ratios.append(ratio)
    return {'passed':True,'native_requests':1152,'identical_request_payloads':True,'natural_outputs_identical':True,
            'cohorts':cohorts,'pooled_points':pooled,'throughput_ratios':ratios,'proofs':hashes,
            'suite_audit_sha256':sha(suitefile),'original_coordinator_passed':False,'original_failure':'R84 audit compared disabled null directly with false; original failed marker retained, full original data rechecked by R105 with explicit null/false role validation and launch/model-path checks.','terminal_marker_sha256':sha(marker),'source_sha256':sha(Path(__file__)),
            'scope':'Complete R84 forward/reverse mixed policies on GPU2/R55 at equal KV65536/SSM256/mem.52. Six measured batches per policy/cache; pooled rates and request latency distributions reconstructed from native original SSE. Concurrent queues, nominal cache labels and small repeat count must be considered. No confidence interval, measured bandwidth, isolated-machine, larger topology, real prefix-hit or overall goal acceptance. The full R105 recheck retains all R84 precision/lifecycle/prefix gates and first-three kernel profiles; no new profile inferred for reverse controls.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'mixed-six-comparison-r104.json';assert not out.exists()
    result=compare(a.root);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['passed','native_requests','pooled_points','throughput_ratios']}))
