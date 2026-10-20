"""Compare actual TP8/DP8 resident KV at identical tensor-byte budgets.

CPU-only; requires both original independent capacity audits and reparses their
raw same-scrape metrics. No summed per-worker peaks or whole-GPU maximum claim.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from audit_attention_dp8_capacity_r135 import (
    IncompleteSnapshot, normalize_snapshot, parse_prometheus, per_rank,
)

IMAGE = 'sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def reconstruct(root, topology):
    log_suffix = 'r31' if topology == 'tp8' else 'r135'
    data_suffix = 'r63' if topology == 'tp8' else 'r135'
    # The R64 capped-KV rerun stores data in r63 and keeps the r31 container/log.
    audit_suffix = 'r64' if topology == 'tp8' else 'r135'
    report_path = root / f'capacity-{topology}-independent-audit-{audit_suffix}.json'
    report = read(report_path)
    assert report['passed'] and report['native_requests_audited'] == 13
    assert report['actual_container_image'] == IMAGE
    assert report['actual_physical_devices'] == list(map(str,range(8)))
    assert report['full_attention_KV_bytes_per_rank'] == 536870912
    assert set(map(int,report['all_eight_rank_allocation_logs'])) == set(range(8))
    placement_file = root / ('actual-placement-observations-r92.jsonl' if topology == 'tp8' else 'actual-placement-observations-r135.jsonl')
    placements = [json.loads(line) for line in placement_file.read_text().splitlines()]
    name = f'qwen38-{topology}-capacity-{log_suffix}'
    matches = [v for v in placements if v['name'] == name]; assert len(matches) == 1
    placement = matches[0]
    assert placement['placement_matched'] and placement['image'] == IMAGE
    assert placement['observed_compute_gpus'] == placement['expected_compute_gpus'] == list(range(8))
    assert len({v['pid'] for v in placement['scheduler_gpu_processes']}) == 8
    directory = root / f'capacity-{topology}-{data_suffix}'
    for name, value in report['files_sha256'].items(): assert sha(directory/name) == value, name
    log = root / f'qwen38-{topology}-capacity-{log_suffix}-final.log'
    assert sha(log) == report['allocation_log_sha256']
    config_doc = read(directory/'server-info.json'); config = config_doc.get('server_args',config_doc)
    assert config['tp_size'] == 8 and config['attn_dp_size'] == (8 if topology == 'dpa8' else 1)
    assert config['dcp_size'] == 1 and config['page_size'] == 1
    assert config['max_mamba_cache_size'] == 256
    summary = read(directory/'summary.json'); assert summary['all_outputs_complete'] and not summary['metric_errors']
    raw = []
    with gzip.open(directory/'raw-metrics.jsonl.gz','rt') as stream:
        for line in stream:
            row = json.loads(line)
            metrics = parse_prometheus(row['raw_prometheus'])
            memory = per_rank(metrics,'kv_cache_memory_usage_gb',set(range(8)))
            assert all(math.isclose(v*1024**3,536870912,abs_tol=.1,rel_tol=0) for v in memory.values())
            raw.append((row,metrics))
    results = []; payloads = {}
    for case in summary['cases']:
        count = case['http_concurrency']; assert count in (1,4,8)
        proof = report['raw_payload_hashes'][str(count)]
        p = directory/f'payload-c{count}.json'; responses = directory/f'responses-c{count}.json'
        assert sha(p) == proof['payload_sha256'] and sha(responses) == proof['responses_sha256']
        payloads[count] = read(p)
        assert len(payloads[count]['input_ids']) == count
        assert all(len(v) == 6000 for v in payloads[count]['input_ids'])
        assert len({tuple(v[:8]) for v in payloads[count]['input_ids']}) == count
        outputs = read(responses); assert len(outputs) == count
        for output in outputs:
            assert len(output['output_ids']) == 512
            assert output['meta_info'].get('cached_tokens',0) == 0
            assert output['meta_info']['prompt_tokens'] == 6000
            assert output['meta_info']['completion_tokens'] == 512
        samples = []
        for row,metrics in raw:
            a,b = row['observation_started_monotonic'],row['monotonic']
            if not case['started_monotonic'] <= a <= b <= case['ended_monotonic']: continue
            try: normalized = normalize_snapshot(metrics,8 if topology == 'dpa8' else 1)
            except IncompleteSnapshot: continue
            samples.append({'monotonic':b,'raw_sha256':hashlib.sha256(row['raw_prometheus'].encode()).hexdigest(),
                            'normalized':normalized})
        assert samples
        peak = max(samples,key=lambda s:s['normalized']['totals']['kv_used_tokens'])
        saved = next(c for c in report['cases'] if c['http_concurrency'] == count)
        assert json.loads(json.dumps(peak)) == saved['peak_same_scrape_proof']
        full = [s for s in samples if s['normalized']['totals']['num_running_reqs'] == count
                and s['normalized']['totals']['kv_used_tokens'] >= 6000*count
                and s['normalized']['totals']['decode_sum_seq_lens'] >= 6000*count]
        sustained = any(0 < b['monotonic']-a['monotonic'] < 1.5 for a,b in zip(full,full[1:]))
        assert sustained == saved['full_requested_batch_observed_in_multiple_scrapes']
        results.append({'http_concurrency':count,'peak':peak,'sustained_full_batch':sustained,
                        'full_batch_samples':len(full),'eligible_samples':len(samples)})
    assert [r['http_concurrency'] for r in results] == [1,4,8]
    return {'original_report_sha256':sha(report_path),'allocation_log_sha256':sha(log),
            'actual_placement':placement,
            'cases':results,'configuration':config,'payloads':payloads,
            'logical_configured_rows':report['logical_capacity_per_attention_worker']*report['attention_workers'],
            'raw_snapshot_count':len(raw),'raw_files_sha256':report['files_sha256']}


def gain(left, right):
    a = next(c for c in left['cases'] if c['http_concurrency'] == 8)
    b = next(c for c in right['cases'] if c['http_concurrency'] == 8)
    observed_a = a['peak']['normalized']['totals']['kv_used_tokens']
    observed_b = b['peak']['normalized']['totals']['kv_used_tokens']
    assert observed_a > 0 and observed_b > 0
    # Positive acceptance requires actual eight-request residency, not just completion.
    accepted = b['sustained_full_batch'] and observed_b >= 48000 and observed_b > observed_a
    return {'passed':accepted,'TP8_c8_peak_resident_tokens':observed_a,
            'DP8_c8_peak_resident_tokens':observed_b,'observed_peak_ratio':observed_b/observed_a,
            'TP8_c8_full_batch_sustained':a['sustained_full_batch'],
            'DP8_c8_full_batch_sustained':b['sustained_full_batch'],
            'configured_logical_row_ratio':right['logical_configured_rows']/left['logical_configured_rows']}


def main(root, output):
    assert not output.exists()
    source = read(root/'source-kv-gain-r150.json')
    for name,digest in source['source_files'].items(): assert sha(root/name) == digest,name
    left,right = (reconstruct(root,top) for top in ('tp8','dpa8'))
    assert left['payloads'] == right['payloads'], 'Input tokens and sampling controls must match'
    for key in ('dtype','kv_cache_dtype','context_length','page_size','max_running_requests',
                'max_mamba_cache_size','mem_fraction_static','enable_deterministic_inference',
                'attention_backend','linear_attn_prefill_backend','mamba_ssm_dtype'):
        assert left['configuration'].get(key) == right['configuration'].get(key),key
    result = {'full_goal_complete':False,**gain(left,right),
              'runtime_image':IMAGE,'full_attention_KV_bytes_per_rank':536870912,
              'physical_GPU_count':8,'native_requests_reparsed':26,
              'controls':{'TP8':left,'DP8':right},'source_manifest_sha256':sha(root/'source-kv-gain-r150.json'),
              'scope':'Same fixed4GiB aggregate full-attention KV tensor budget and physicalGPUs0..7. Raw same-scrape observations/unique6000-input+512-output/native completion and original audit hashes checked. Accepted gain requires sustained8-request DP8 residency and a strictly larger observed c8 peak than TP8; configured-row ratio is separate. Gauges update asynchronously; not maximum whole-GPU capacity, per-request context limit, task quality, cross-topology numerical parity or performance acceptance. SSM/metadata and model memory are outside the512MiB/rank full-attention KV budget.'}
    with output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='controls'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
