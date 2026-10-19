"""Independent original P/D payload, natural SSE, percentile and rank-trace audit.

P HTTP elapsed includes prefill/transfer and is not pure communication time.
Memcpy totals are profile-wide, not proof of transport-only staging costs.
"""
import argparse,collections,gzip,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b):assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def quantile(v,q):return sorted(v)[math.ceil(len(v)*q)-1]

def native(directory,side,tp,smoke=False):
    rows=lines(directory/'requests.jsonl');groups=lines(directory/'groups.jsonl');summary=read(directory/'summary.json')
    expected_requests=2 if smoke else 636;rounds=1 if smoke else 3
    assert len(rows)==len({r['label'] for r in rows})==summary['request_count']==expected_requests
    assert len(groups)==summary['groups']==(2 if smoke else 48)
    hashes={};records={};rooms=set()
    for row in rows:
        p=directory/(row['label']+'.json');before=sha(p);raw=read(p)
        req=raw['decode_request'];pre=raw['prefill'];preq=pre['request'];pres=pre['response'];events=raw['decode_events'];nout=row['output_tokens'];nin=row['input_tokens']
        assert req['stream'] is True and preq['stream'] is False
        assert 'custom_logit_processor' not in req and 'custom_logit_processor' not in preq
        comparable=lambda r:{k:v for k,v in r.items() if k!='stream'}
        assert comparable(req)==comparable(preq)
        room=req['bootstrap_room'];assert type(room) is int and 0<=room<2**63 and room not in rooms;rooms.add(room)
        assert req['bootstrap_host']=='127.0.0.1' and req['bootstrap_port']>0
        assert len(req['input_ids'])==nin and req['sampling_params']=={'temperature':0,'max_new_tokens':nout,'ignore_eos':True}
        assert req.get('lora_path')==('perf-A' if side=='d' else None)
        first=last=None;previous=[];prev=0;clock=-1;first_count=0
        for event in events:
            now=event['elapsed_s'];data=event['data'];ids=data['output_ids'];count=len(ids)
            assert math.isfinite(now) and now>=clock and count>=prev and ids[:prev]==previous
            if count>prev:
                if first is None:first,first_count=now,count
                last=now
            previous,prev,clock=ids,count,now
        assert first is not None and 0<first_count<nout and prev==nout and previous==row['output_ids']
        assert pres['output_ids']==previous[:1] and pres['meta_info']['completion_tokens']==1
        assert pres['meta_info']['prompt_tokens']==data['meta_info']['prompt_tokens']==nin
        assert data['meta_info']['completion_tokens']==nout and data['meta_info']['finish_reason']=={'type':'length','length':nout}
        assert row['first_nonempty_event_tokens']==first_count
        same(row['ttft_s'],first);same(row['tpot_s'],(last-first)/(nout-first_count))
        same(row['prefill_http_elapsed_s'],pre['http_elapsed_s'])
        assert math.isfinite(row['e2e_s']) and row['e2e_s']>=max(last,pre['http_elapsed_s'])
        assert row['cached_tokens']==pres['meta_info'].get('cached_tokens')
        records[row['label']]={'input_ids':req['input_ids'],'sampling_params':req['sampling_params'],'output_ids':previous}
        hashes[p.name]=before;assert sha(p)==before
    nin_values=[(128,16)] if smoke else [(128,128),(2048,256)];cs=[1] if smoke else [1,4,16,32]
    labels={f'in{nin}-out{nout}-c{c}-r{r}-{cache}-i{i}' for nin,nout in nin_values for c in cs for r in range(rounds) for cache in ['cold','hot'] for i in range(c)}
    assert set(records)==labels
    seen=set()
    for group in groups:
        key=(group['input_tokens'],group['concurrency'],group['round'],group['cache']);assert key not in seen;seen.add(key)
        batch=[r for r in rows if (r['input_tokens'],r['concurrency'],r['round'],r['cache'])==key]
        assert len(batch)==group['concurrency'] and group['elapsed_s']>=max(r['e2e_s'] for r in batch)
        same(group['output_throughput_tokens_s'],sum(r['output_tokens'] for r in batch)/group['elapsed_s'])
    assert seen=={(nin,c,r,cache) for nin,_ in nin_values for c in cs for r in range(rounds) for cache in ['cold','hot']}
    assert len(summary['points'])==(2 if smoke else 16)
    point_keys=set();stats_checked=0;points=[]
    for point in summary['points']:
        key=(point['input_tokens'],point['concurrency'],point['cache']);assert key not in point_keys;point_keys.add(key)
        rr=[r for r in rows if (r['input_tokens'],r['concurrency'],r['cache'])==key];gg=[g for g in groups if (g['input_tokens'],g['concurrency'],g['cache'])==key]
        assert len(rr)==point['requests']==rounds*key[1] and len(gg)==rounds
        same(point['throughput_tokens_s_mean'],math.fsum(g['output_throughput_tokens_s'] for g in gg)/rounds)
        for metric in ['ttft_s','tpot_s','e2e_s']:
            values=[r[metric] for r in rr];calculated={'mean':math.fsum(values)/len(values),**{f'p{int(q*100)}':quantile(values,q) for q in [.5,.95,.99]}}
            for k,v in calculated.items():same(v,point[metric][k]);stats_checked+=1
        hits=[r['cached_tokens'] for r in rr if r['cached_tokens'] is not None]
        points.append({**point,'actual_cached_tokens_min_max':[min(hits),max(hits)] if hits else None})
    configs={}
    for role,f,gpu in [('prefill','prefill-server-info.json',0),('decode','server-info.json',tp)]:
        x=read(directory/f);cfg=x.get('server_args',x);configs[role]=cfg
        assert cfg['tp_size']==tp and cfg['attn_dp_size']==1 and cfg['base_gpu_id']==gpu
        assert cfg['disaggregation_mode']==role and cfg['disaggregation_transfer_backend']=='nixl'
        assert cfg['disaggregation_enable_kv_checksum'] and cfg['mamba_radix_cache_strategy']=='extra_buffer'
        assert bool(cfg['enable_lora'])==(side=='d')
    return {'passed':True,'requests':len(rows),'groups':len(groups),'speed_points':points,'latency_statistics_recomputed':stats_checked,'original_payloads_sha256':hashes,
            'summary_sha256':sha(directory/'summary.json'),'server_configurations':configs,
            'scope':'Original unforced P/D request pairing and natural SSE timestamps checked; E2E is retained client completion elapsed checked against both P HTTP and last-SSE lower bounds; actual prefix hits reported.'},records


def profiles(directory,tp):
    saved=read(directory/'kernel-summary.json');listed={Path(t['file']).name:t for t in saved['traces']};result={}
    for role in ['prefill','decode']:
        paths=sorted((directory/'profile'/role).glob('*.trace.json.gz'));assert len(paths)==tp
        ranks=[];reports=[]
        for p in paths:
            h=sha(p);assert h==listed[p.name]['sha256']
            with gzip.open(p,'rt') as f:doc=json.load(f)
            kernels=[e for e in doc['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X' and isinstance(e.get('dur'),(int,float))]
            assert len(kernels)==listed[p.name]['gpu_kernel_count']>0
            assert all(math.isfinite(e['dur']) and e['dur']>=0 for e in kernels)
            names=collections.Counter(e['name'] for e in kernels);copies=collections.Counter(e['name'] for e in doc['traceEvents'] if e.get('cat')=='gpu_memcpy' and e.get('ph')=='X')
            rank=int(p.name.split('-TP-')[1].split('.')[0].split('-')[0]);ranks.append(rank)
            reports.append({'rank':rank,'trace':p.name,'sha256':h,'bytes':p.stat().st_size,'kernel_events':len(kernels),
                            'reported_GPU_devices':listed[p.name]['gpu_devices'],'FP32_sgemm_events':sum(n for k,n in names.items() if 'sgemm' in k.lower()),
                            'BF16_merge_store_events':sum(n for k,n in names.items() if '_round_merged_weight' in k),
                            'checksum_kernel_events':sum(n for k,n in names.items() if 'adler32' in k.lower()),
                            'collective_kernel_counts':{k:n for k,n in names.items() if any(v in k.lower() for v in ['nccl','allreduce','all_reduce','cross_device','reduce_scatter','all_gather'])},
                            'profile_wide_memcpy_counts':dict(copies),'all_exact_kernel_counts':dict(names)})
            del doc
        assert set(ranks)==set(range(tp)) and len(ranks)==tp
        result[role]=reports
    assert len(listed)==tp*2
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--diagnostic-native-only',action='store_true');a=p.parse_args();root=a.root
    if a.diagnostic_native_only:
        report,_=native(root/'perf-pd2diag-m-r80','m',2,True)
        result={'passed':True,'scope':'Existing2P2D M-only two-request native smoke parser replay; original P profile failed, not a completed PD experiment or full4P4D acceptance.','native':report,'full4P4D_accepted':False}
        filename='pd2-native-parser-audit-r95.json'
    else:
        assert read(root/'pd4-precision-perf-complete-r33.json')['natural_outputs_equal']
        controls={};records={};kernel_profiles={}
        for side in ['d','m','base']:
            directory=root/f'perf-pd4-{side}-r33';complete=read(directory/'complete.json');assert complete['requests']==636 and complete['profile'] and not complete['smoke']
            controls[side],records[side]=native(directory,side,4)
            kernel_profiles[side]=profiles(directory,4)
            if side!='d':assert all(r['FP32_sgemm_events']==r['BF16_merge_store_events']==0 for rs in kernel_profiles[side].values() for r in rs)
        assert records['d']==records['m']
        for key,r in records['m'].items():assert all(r[k]==records['base'][key][k] for k in ['input_ids','sampling_params'])
        result={'passed':True,'scope':'Full4P4D three-way natural native timing and actual per-role/per-rank profile audit; separate source/precision/physical-placement/capacity gates remain mandatory. Memcpy counts cover whole profile, not isolated NIXL transfer bandwidth.','native_requests':1908,'speed_points':48,'latency_statistics_recomputed':sum(c['latency_statistics_recomputed'] for c in controls.values()),'natural_D_M_outputs_equal':True,'controls':controls,'actual_profile_traces':kernel_profiles}
        assert result['latency_statistics_recomputed']==576
        filename='pd4-native-kernel-audit-r95.json'
    result['source_sha256']=sha(Path(__file__));out=root/filename;assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['controls','native','actual_profile_traces']}))
if __name__=='__main__':main()
