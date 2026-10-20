"""Read-only, labeled, same-scrape audit of the future capped KV experiment.

The raw sampler's summed metric_peaks are deliberately never used here.
"""
import argparse
import gzip
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

METRICS = {
    'kv_used_tokens', 'kv_available_tokens', 'kv_evictable_tokens',
    'num_running_reqs', 'decode_sum_seq_lens', 'num_queue_reqs',
    'max_total_num_tokens', 'kv_cache_memory_usage_gb',
    'mamba_used_tokens', 'mamba_available_tokens', 'mamba_evictable_tokens',
}
CORE = ('kv_used_tokens', 'num_running_reqs', 'decode_sum_seq_lens')
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$')
LABEL = re.compile(r'([a-zA-Z_]\w*)="((?:\\.|[^"\\])*)"')

class IncompleteSnapshot(ValueError):
    """An idle/reporting worker has not emitted a required series yet."""

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_labels(text):
    result={};pos=0
    while pos < len(text):
        m=LABEL.match(text,pos)
        assert m, ('Malformed Prometheus labels',text,pos)
        key=m[1];assert key not in result
        result[key]=json.loads('"'+m[2]+'"');pos=m.end()
        if pos < len(text):
            assert text[pos]==','
            pos+=1
    return result

def parse_prometheus(raw):
    rows=[];seen=set()
    for line in raw.splitlines():
        m=SAMPLE.match(line)
        if not m or not m[1].startswith('sglang:') or m[1][7:] not in METRICS:
            continue
        labels=parse_labels(m[2] or '')
        if labels.get('priority','')!='':
            continue
        value=float(m[3]);assert math.isfinite(value) and value>=0,(m[1],value)
        key=(m[1],tuple(sorted(labels.items())))
        assert key not in seen, ('Duplicate metric series',key)
        seen.add(key)
        rows.append({'metric':m[1][7:],'labels':labels,'value':value})
    return rows

def per_rank(rows, metric, expected_ranks):
    values={}
    for row in rows:
        if row['metric']!=metric:
            continue
        labels=row['labels'];assert int(labels.get('pp_rank',0))==0
        rank=int(labels['tp_rank'])
        assert 0 <= rank < 8
        if rank not in expected_ranks:
            continue
        assert rank not in values, ('Ambiguous rank series',metric,rank)
        values[rank]=row['value']
    if set(values)!=set(expected_ranks):
        raise IncompleteSnapshot(('Missing reporting ranks',metric,set(values),expected_ranks))
    return values

def normalize_snapshot(rows, attn_dp_size):
    assert attn_dp_size in (1,8)
    # R28 source emits active scheduler stats on attn_tp_rank0. DCP leaves
    # this attention-TP layout unchanged; DP8 has one such rank per worker.
    leaders=set(range(0,8,8//attn_dp_size))
    core={metric:per_rank(rows,metric,leaders) for metric in CORE}
    total={metric:sum(values.values()) for metric,values in core.items()}
    return {'leaders':sorted(leaders),'per_worker':core,'totals':total}

def peak_same_scrape(samples):
    assert samples
    return max(samples,key=lambda x:x['normalized']['totals']['kv_used_tokens'])

def audit(top, root):
    directory=root/f'capacity-{top}-r135'
    summary=json.loads((directory/'summary.json').read_text())
    config_doc=json.loads((directory/'server-info.json').read_text())
    config=config_doc.get('server_args',config_doc)
    dp=8 if top=='dpa8' else 1;dcp=2 if top=='dcp8x2' else 1
    rows_cap=8191 if dp==8 else 32767
    assert config['tp_size']==8 and config['attn_dp_size']==dp
    assert config['dcp_size']==dcp and config['page_size']==1
    assert config['context_length']==16384 and config['enable_metrics']
    assert config['enable_lora'] and config['max_running_requests']==32
    assert config['max_mamba_cache_size']==256
    assert summary['all_outputs_complete'] and not summary['metric_errors']
    assert [c['http_concurrency'] for c in summary['cases']]==[1,4,8]
    container_name=f'qwen38-{top}-capacity-r135'
    actual=json.loads(subprocess.check_output(['docker','inspect',container_name],text=True))[0]
    assert actual['Image']=='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
    devices=[v for request in actual['HostConfig']['DeviceRequests'] for v in request.get('DeviceIDs',[])]
    assert devices==list(map(str,range(8))),devices
    assert not actual['State']['Running'] and not actual['State']['OOMKilled']
    log=root/f'qwen38-{top}-capacity-r135-final.log'
    allocations={}
    pat=re.compile(r'TP(\d+)\].*KV Cache is allocated\. dtype: torch.bfloat16, #tokens: (\d+), K size: ([0-9.]+) GB, V size: ([0-9.]+) GB')
    for line in log.read_text().splitlines():
        m=pat.search(line)
        if m:
            rank=int(m[1]);assert rank not in allocations
            assert int(m[2])==rows_cap and float(m[3])==float(m[4])==.25
            allocations[rank]={'physical_rows':int(m[2]),'K_GiB_rounded':float(m[3]),'V_GiB_rounded':float(m[4])}
    assert set(allocations)==set(range(8)), 'Need actual allocation logs for all eight ranks'
    parsed={};raw_hashes={}
    with gzip.open(directory/'raw-metrics.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);stamp=row['monotonic'];assert stamp not in parsed
            assert row['observation_started_monotonic'] <= stamp
            raw_hashes[stamp]=hashlib.sha256(row['raw_prometheus'].encode()).hexdigest()
            parsed[stamp]={'row':row,'metrics':parse_prometheus(row['raw_prometheus'])}
    captured=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    assert len(captured)==len(parsed)
    constants_verified=0
    for row in captured:
        stamp=row['monotonic'];raw=parsed[stamp]
        assert row['raw_sha256']==raw_hashes[stamp]
        assert row['observation_started_monotonic']==raw['row']['observation_started_monotonic']
        # This gauge's actual R28 owner is HybridLinearKVPool's full MHA
        # child; SSM lives in ReqToTokenPool and is not included in this gauge.
        memory=per_rank(raw['metrics'],'kv_cache_memory_usage_gb',set(range(8)))
        assert all(math.isclose(v*1024**3,536870912,rel_tol=0,abs_tol=.1) for v in memory.values()),memory
        constants_verified+=1
    cases=[];payload_proofs={};completed=0
    for case in summary['cases']:
        count=case['http_concurrency'];payload_file=directory/f'payload-c{count}.json'
        payload=json.loads(payload_file.read_text());outputs_file=directory/f'responses-c{count}.json'
        outputs=json.loads(outputs_file.read_text())
        assert set(payload)=={'input_ids','lora_path','sampling_params'}
        assert len(payload['input_ids'])==len(outputs)==count
        assert all(len(x)==6000 for x in payload['input_ids'])
        assert len({tuple(x[:8]) for x in payload['input_ids']})==count
        assert payload['lora_path']==['capacity-A']*count
        assert payload['sampling_params']==[{'temperature':0,'max_new_tokens':512,'ignore_eos':True}]*count
        for output in outputs:
            assert len(output['output_ids'])==512
            meta=output['meta_info'];assert meta['prompt_tokens']==6000 and meta['completion_tokens']==512
            assert meta['finish_reason']['type']=='length'
            assert meta.get('cached_tokens',0)==0, 'Unique prompts after flush must not reuse KV'
        started=case['started_monotonic'];ended=case['ended_monotonic'];assert started<ended
        assert math.isclose(ended-started,case['elapsed_s'],abs_tol=1e-9)
        eligible=[];incomplete=[]
        for stamp,item in parsed.items():
            raw=item['row']
            if not (started<=raw['observation_started_monotonic']<=stamp<=ended):continue
            try:norm=normalize_snapshot(item['metrics'],dp)
            except IncompleteSnapshot as e:
                incomplete.append({'monotonic':stamp,'reason':str(e)});continue
            used=norm['totals']['kv_used_tokens'];running=norm['totals']['num_running_reqs']
            assert 0<=running<=count and used<=rows_cap*dcp*dp
            eligible.append({'monotonic':stamp,'normalized':norm,'raw_sha256':raw_hashes[stamp]})
        assert eligible, ('No complete same-time worker snapshot',top,count,incomplete[:3])
        peak=peak_same_scrape(eligible)
        # A sustained full-batch observation is stronger than a single scrape.
        full=[s for s in eligible if s['normalized']['totals']['num_running_reqs']==count
              and s['normalized']['totals']['kv_used_tokens']>=6000*count
              and s['normalized']['totals']['decode_sum_seq_lens']>=6000*count]
        sustained=any(0 < b['monotonic']-a['monotonic'] < 1.5
                      for a,b in zip(full,full[1:]))
        cases.append({'http_concurrency':count,'completed_native_requests':count,
                      'eligible_same_scrape_samples':len(eligible),'incomplete_samples':len(incomplete),
                      'peak_observed_logical_KV_tokens':peak['normalized']['totals']['kv_used_tokens'],
                      'running_requests_at_that_same_peak':peak['normalized']['totals']['num_running_reqs'],
                      'peak_same_scrape_proof':peak,
                      'max_running_requests_one_scrape':max(s['normalized']['totals']['num_running_reqs'] for s in eligible),
                      'full_requested_unique_prompt_batch_observed':bool(full),
                      'full_requested_batch_observed_in_multiple_scrapes':sustained})
        payload_proofs[str(count)]={'payload_sha256':sha(payload_file),'responses_sha256':sha(outputs_file)}
        completed+=count
    assert completed==13
    report={'passed':True,'topology':top,'native_requests_audited':13,'full_attention_KV_bytes_per_rank':536870912,
            'actual_container_image':actual['Image'],'actual_physical_devices':devices,
            'all_eight_rank_allocation_logs':allocations,'memory_gauge_snapshots_verified':constants_verified,
            'physical_rows_per_rank':rows_cap,'logical_capacity_per_attention_worker':rows_cap*dcp,
            'attention_workers':dp,'cases':cases,'raw_payload_hashes':payload_proofs,
            'files_sha256':{name:sha(directory/name) for name in ['summary.json','server-info.json','raw-metrics.jsonl.gz','metrics.jsonl']},
            'allocation_log_sha256':sha(log),
            'scope':'Candidate R116 image, new R118 namespace; original R64 same-scrape audit retained. Observed lower bounds at a fixed full-attention KV tensor budget, not maximum whole-GPU capacity or long-context quality. R28 leader-rank series only; TP duplicate constants are never summed. Each reported peak is from one labeled scrape wholly inside the request window. Prometheus gauges update asynchronously; multiple full-batch observations are separately reported. Native output completion is checked; independent D/M accuracy is separate. SSM and auxiliary metadata are separate from the 512MiB full-attention tensor budget.'}
    target=root/f'capacity-{top}-independent-audit-r135.json';assert not target.exists()
    target.write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('topology',choices=('tp8','dcp8x2','dpa8'))
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    result=audit(a.topology,a.root)
    print(json.dumps({'passed':result['passed'],'topology':result['topology'],'cases':result['cases']}))
