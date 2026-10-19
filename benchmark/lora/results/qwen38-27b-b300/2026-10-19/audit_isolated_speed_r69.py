"""Validate all isolated speed raw artifacts before publishing a ratio."""
import hashlib
import json
import subprocess
from pathlib import Path
R = Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
order = ['nocache', 'cached', 'm', 'base', 'cached', 'nocache']
resolved = json.loads((R / 'isolated-resolved-configs-r69.json').read_text())
assert len(resolved) == len(order) == 6
keys = ['tp_size', 'dp_size', 'attn_dp_size', 'pp_size', 'dtype', 'kv_cache_dtype',
        'context_length', 'max_req_input_len', 'chunked_prefill_size', 'max_running_requests',
        'max_total_num_tokens', 'max_mamba_cache_size', 'mem_fraction_static',
        'attention_backend', 'linear_attn_backend', 'linear_attn_prefill_backend',
        'linear_attn_decode_backend', 'bf16_gemm_backend', 'enable_deterministic_inference',
        'disable_cuda_graph', 'mamba_ssm_dtype', 'mamba_radix_cache_strategy', 'page_size']
basecfg = resolved[0]['config']
assert basecfg['mem_fraction_static'] == .52
assert basecfg['max_total_num_tokens'] == 65536 and basecfg['max_mamba_cache_size'] == 256
measurements, sources, first_times, last_times = [], {}, [], []
for position, side in enumerate(order):
    item = resolved[position]
    assert (item['position'], item['side']) == (position, side)
    assert not {key: (basecfg.get(key), item['config'].get(key)) for key in keys if basecfg.get(key) != item['config'].get(key)}
    expected_cache = 36864 if side == 'cached' else 0
    assert item['config']['lora_merged_bf16_cache_size_mb'] == expected_cache
    launch = json.loads((R / f'launch-qwen38-isolated-{position}-{side}-r69.json').read_text())
    assert launch[launch.index('--gpus') + 1] == '"device=2"' and 'sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7' in launch
    actual=json.loads(subprocess.check_output(['docker','inspect',f'qwen38-isolated-{position}-{side}-r69'],text=True))[0]
    assert actual['Image']=='sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7'
    assert not actual['State']['Running'] and not actual['State']['OOMKilled']
    d = R / f'perf-isolated-{position}-{side}-r69'
    summary = json.loads((d / 'summary.json').read_text())
    rows = [json.loads(line) for line in (d / 'requests.jsonl').read_text().splitlines()]
    groups = [json.loads(line) for line in (d / 'groups.jsonl').read_text().splitlines()]
    assert summary['request_count'] == len(rows) == len({row['label'] for row in rows}) == 636
    assert summary['groups'] == len(groups) == 48 and len(summary['points']) == 16
    assert json.loads((d / 'complete.json').read_text()) == {'requests': 636, 'profile': False}
    request_shapes = {}
    for row in rows:
        native = json.loads((d / (row['label'] + '.json')).read_text())
        req, events = native['request'], native['events']
        assert req['stream'] and req['sampling_params'] == {'temperature': 0, 'max_new_tokens': row['output_tokens'], 'ignore_eos': True}
        assert ('lora_path' in req) == (side in ['cached', 'nocache'])
        nonempty = [event for event in events if event['data'].get('output_ids')]
        assert nonempty and len(nonempty[0]['data']['output_ids']) == 1, 'Original client TPOT requires one token in its first event'
        assert events[-1]['data']['output_ids'] == row['output_ids'] and len(row['output_ids']) == row['output_tokens']
        assert len(req['input_ids']) == row['input_tokens']
        assert abs(nonempty[0]['elapsed_s'] - row['ttft_s']) < 1e-5
        assert abs((nonempty[-1]['elapsed_s'] - nonempty[0]['elapsed_s']) / (row['output_tokens'] - 1) - row['tpot_s']) < 1e-5
        end = (d / (row['label'] + '.json')).stat().st_mtime
        first_times.append(end - row['e2e_s']); last_times.append(end)
        request_shapes[row['label']] = req['input_ids']
    for file in ['summary.json', 'requests.jsonl', 'groups.jsonl', 'server-info.json', 'complete.json']:
        sources[str((d / file).relative_to(R))] = hashlib.sha256((d / file).read_bytes()).hexdigest()
    measurements.append({'position': position, 'side': side, 'summary': summary,
                         'rows': {row['label']: row['output_ids'] for row in rows}, 'inputs': request_shapes})
reference = measurements[2]
for record in measurements:
    assert record['inputs'] == reference['inputs'], record['position']
    if record['side'] != 'base':
        assert record['rows'] == reference['rows'], record['position']
monitor = [json.loads(line) for line in (R / 'isolated-speed-gpu-observations-r69.jsonl').read_text().splitlines() if line.strip()]
assert len(monitor) >= 10 and monitor[0]['unix_s'] <= min(first_times) and monitor[-1]['unix_s'] >= max(last_times) - 2
assert not any(record['unexpected_active_gpus'] for record in monitor)
assert all(len(record['gpus']) == 8 and all(gpu['memory_mib'] < 2000 and int(gpu['utilization_pct']) == 0 for gpu in record['gpus'] if gpu['index'] != 2) for record in monitor)
fill = []
for position, side in enumerate(order[:4]):
    d = R / f'fill-isolated-{position}-{side}-r69'
    rows = [json.loads(line) for line in (d / 'requests.jsonl').read_text().splitlines()]
    summary = json.loads((d / 'summary.json').read_text())
    assert len(rows) == summary['request_count'] == 10 and summary['cycles'] == 5
    assert all(row['cached_tokens'] == 0 and row['output_tokens'] == 16 and row['input_tokens'] == 1630 for row in rows)
    assert all(row['output_ids'] == rows[0]['output_ids'] for row in rows)
    fill.append({'position': position, 'side': side, 'summary': summary, 'output_ids': rows[0]['output_ids']})
    for file in ['summary.json', 'requests.jsonl', 'api.json', 'server-info.json']:
        sources[str((d / file).relative_to(R))] = hashlib.sha256((d / file).read_bytes()).hexdigest()
assert fill[0]['output_ids'] == fill[1]['output_ids'] == fill[2]['output_ids']
points = []
for a, b in [(0, 1), (5, 4)]:
    for cold in measurements[a]['summary']['points']:
        key = tuple(cold[k] for k in ['input_tokens', 'concurrency', 'cache'])
        cached = next(row for row in measurements[b]['summary']['points'] if tuple(row[k] for k in ['input_tokens', 'concurrency', 'cache']) == key)
        points.append({'uncached_position': a, 'cached_position': b, 'input_tokens': key[0], 'concurrency': key[1], 'cache': key[2],
                       'uncached': cold, 'cached': cached, 'throughput_ratio': cached['throughput_tokens_s_mean'] / cold['throughput_tokens_s_mean']})
out = {'scope': 'Actual dedicated-machine run; all one-second samples show other GPUs below2GiB and at0% utilization during measurement. Six sequential full matrices, same GPU2/image/KV/SSM; cache-control order reversed for repetition. These are observed measurements, not a universal hardware guarantee. c32*2304>65536 may queue. Cold fill includes prefill/decode and adapter movement, not an isolated memcpy/merge timing.',
       'matrix_runs': 6, 'matrix_requests': 3816, 'fill_requests': 40, 'natural_output_matches_per_nonbase_run': 636,
       'resolved_matching_config': {key: basecfg.get(key) for key in keys}, 'monitor_samples': len(monitor),
       'source_files_sha256': sources, 'points': points,
       'merged': measurements[2]['summary'], 'base': measurements[3]['summary'], 'fill': fill}
(R / 'isolated-speed-comparison-r69.json').write_text(json.dumps(out, indent=2))
print(json.dumps({'runs': 6, 'requests': 3816, 'fill_requests': 40, 'monitor_samples': len(monitor),
                  'c32_cold_ratios': [{'input_tokens': row['input_tokens'], 'uncached_position': row['uncached_position'], 'cached_position': row['cached_position'], 'ratio': row['throughput_ratio']} for row in points if row['concurrency'] == 32 and row['cache'] == 'cold']}))
