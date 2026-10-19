"""Separate adapter registration, cold projection fill, and warm requests.

KV is flushed before both cold and warm requests. Measurements include prefill
and natural decode, so they are not an isolated device-copy/merge timer.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from compare_merged import call, quantile

p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--adapter-path')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
info = call(a.url, '/get_server_info')
(a.output / 'server-info.json').write_text(json.dumps(info, indent=2))
corpus = json.loads(Path('/work/qwen38-mol-speed-20261019/corpus-mixed.json').read_text())
ids = max(corpus, key=lambda row: len(row['input_ids']))['input_ids']
assert len(ids) == 1630
api, rows = [], []

def control(path, body):
    start = time.monotonic()
    result = call(a.url, path, body)
    elapsed = time.monotonic() - start
    api.append({'path': path, 'request': body, 'response': result, 'elapsed_s': elapsed})
    (a.output / 'api.json').write_text(json.dumps(api, indent=2))
    return elapsed

def generate(label):
    control('/flush_cache', {})
    req = {'input_ids': ids, 'stream': True,
           'sampling_params': {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True}}
    if a.adapter_path:
        req['lora_path'] = 'fill-A'
    request = urllib.request.Request(a.url + '/generate', json.dumps(req).encode(),
                                     headers={'Content-Type': 'application/json'})
    events, first, last, first_count, prev = [], None, None, 0, 0
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=3600) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith('data:'):
                continue
            raw = line[5:].strip()
            if raw == '[DONE]':
                break
            result = json.loads(raw)
            now = time.monotonic()
            events.append({'elapsed_s': now - start, 'data': result})
            count = len(result.get('output_ids', []))
            if count > prev:
                if first is None:
                    first, first_count = now, count
                last, prev = now, count
    elapsed = time.monotonic() - start
    (a.output / (label + '.json')).write_text(json.dumps({'request': req, 'events': events}))
    assert first is not None and prev == 16 and first_count < 16
    final = events[-1]['data']
    assert final['meta_info'].get('cached_tokens', 0) == 0, final['meta_info']
    return {'label': label, 'input_tokens': len(ids), 'output_tokens': 16,
            'first_event_tokens': first_count, 'ttft_s': first - start,
            'e2e_s': elapsed, 'tpot_s': (last - first) / (16 - first_count),
            'cached_tokens': final['meta_info'].get('cached_tokens', 0),
            'output_ids': final['output_ids']}

loaded = False
try:
    for cycle in range(5):
        registration = None
        if a.adapter_path:
            registration = control('/load_lora_adapter', {'lora_name': 'fill-A', 'lora_path': a.adapter_path})
            loaded = True
        try:
            for mode in ['first_after_load', 'warm_projection']:
                row = generate(f'cycle{cycle}-{mode}')
                row.update(cycle=cycle, mode=mode, registration_s=registration)
                rows.append(row)
                with (a.output / 'requests.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
        finally:
            if loaded:
                control('/unload_lora_adapter', {'lora_name': 'fill-A'})
                loaded = False
    assert len(rows) == 10 and all(row['output_ids'] == rows[0]['output_ids'] for row in rows)
    points = []
    for mode in ['first_after_load', 'warm_projection']:
        selected = [row for row in rows if row['mode'] == mode]
        point = {'mode': mode, 'requests': len(selected)}
        for metric in ['ttft_s', 'e2e_s', 'tpot_s']:
            values = [row[metric] for row in selected]
            point[metric] = {'mean': sum(values) / len(values), 'p50': quantile(values, .5),
                             'p95': quantile(values, .95), 'p99': quantile(values, .99)}
        points.append(point)
    (a.output / 'summary.json').write_text(json.dumps({'request_count': 10, 'cycles': 5, 'points': points,
        'scope': 'Natural 1630-input/16-output requests, c1. Five alias load/unload cycles; KV flushed for every request. First request after loading includes GPU adapter movement, projection fill, prefill and decode. Registration API time is separate. Warm request preserves projection weights but has no KV hit. For M/base there is no adapter registration.'}, indent=2))
finally:
    if loaded:
        control('/unload_lora_adapter', {'lora_name': 'fill-A'})
