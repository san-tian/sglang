"""R55 isolated SSE/percentile/placement/fill reconstruction; no serving requests."""
import ast
import collections
import copy
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path
from audit_active_raw_r65 import inventory
from run_active_isolated_r147 import ORDER, IMAGE

R = Path(__file__).resolve().parent


def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a, b): assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


class Paths:
    def __init__(self, mapping): self.mapping = mapping
    def __truediv__(self, name): return self.mapping.get(name, R / name)


def native(position, side, placements):
    role = side if side in ('m', 'base') else 'd'
    actual_name = f'qwen38-active-isolated-{position}-{side}-r147'
    directory = R / f'perf-active-isolated-{position}-{side}-r147'
    suffix = 'r41' if side == 'base' else 'r30'
    alias = 'launch-qwen38-isolated-' + ('base-r41' if side == 'base' else 'm-real_r16-r30' if side == 'm' else 'd-r30') + '.json'
    tree = ast.parse((R / 'audit_topology_native_r100.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert {n.name for n in functions} == {'read', 'lines', 'sha', 'cfg'}
    loop = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'side'))
    loop.iter = ast.List(elts=[ast.Constant(value=role)], ctx=ast.Load())
    trace_start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'traces' for t in n.targets))
    points_start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For) and ast.unparse(n.target) == 'point')
    assert trace_start < points_start
    loop.body = loop.body[:trace_start] + loop.body[points_start:]
    changes = 0
    for node in ast.walk(loop):
        if not isinstance(node, ast.Assert): continue
        expr = ast.unparse(node.test)
        if expr == "complete['requests'] == 636 and complete['profile']":
            node.test = ast.parse("complete['requests'] == 636 and not complete['profile']", mode='eval').body; changes += 1
        if expr == "config['context_length'] == 4096 and config['mem_fraction_static'] == 0.5":
            node.test = ast.parse("config['context_length'] == 4096 and config['mem_fraction_static'] == 0.52", mode='eval').body; changes += 1
    assert changes == 2
    mapping = {f'perf-isolated-{role}-{suffix}': directory, alias: R / f'launch-{actual_name}.json'}
    ns = {k: globals()[k] for k in ('collections', 'gzip', 'hashlib', 'json', 'math', 'Path')}
    ns.update(R=Paths(mapping), top='isolated', legacy=True, tp=1, dpa=False, dcp=1,
              expected_gpus='"device=2"', IMAGE=IMAGE, latency_statistics=0)
    for name in ('configurations', 'rows_by_side', 'proofs', 'traces_by_side', 'summaries', 'launches', 'placement_proofs'): ns[name] = {}
    module = ast.fix_missing_locations(ast.Module(body=functions + [loop], type_ignores=[]))
    exec(compile(module, 'R100:R147-no-profile-isolated', 'exec'), ns)
    assert ns['latency_statistics'] == 192
    cfg = ns['configurations'][role]
    assert cfg['max_total_num_tokens'] == 65536 and cfg['max_mamba_cache_size'] == 256
    assert cfg['lora_merged_bf16_cache_size_mb'] == (36864 if side in ('lru', 'active') else 0)
    if side in ('lru', 'active'): assert cfg['lora_merged_bf16_cache_policy'] == side
    command = read(mapping[alias]); assert IMAGE in command
    expected_model = '/work/qwen38-mol-development-20261019/merged/real_r16' if side == 'm' else '/work/models/Qwen3.8-27B/' if side == 'base' else '/work/models/Qwen3.8-27B'
    assert command[command.index('--model-path') + 1] == expected_model
    actual = read(R / f'actual-final-{actual_name}.json')
    assert actual['Image'] == IMAGE and not actual['State']['Running'] and not actual['State']['OOMKilled']
    assert actual['Config']['Cmd'] == command[command.index(IMAGE) + 1:]
    placement = [v for v in placements if v['name'] == actual_name]; assert len(placement) == 1
    placement = placement[0]
    assert placement['placement_matched'] and placement['image'] == IMAGE
    assert placement['expected_compute_gpus'] == placement['observed_compute_gpus'] == [2]
    assert len({v['pid'] for v in placement['scheduler_gpu_processes']}) == 1
    return {'configuration': cfg, 'summary': ns['summaries'][role], 'records': ns['rows_by_side'][role],
            'raw_native_sha256': ns['proofs'][role], 'actual_placement': placement,
            'audit_ast_sha256': hashlib.sha256(ast.dump(module).encode()).hexdigest()}


def fill(position, side):
    directory = R / f'fill-active-isolated-{position}-{side}-r147'
    rows = lines(directory / 'requests.jsonl'); summary = read(directory / 'summary.json')
    assert len(rows) == summary['request_count'] == 10 and summary['cycles'] == 5
    assert {(r['cycle'], r['mode']) for r in rows} == {(c, m) for c in range(5) for m in ('first_after_load', 'warm_projection')}
    for row in rows:
        raw = read(directory / (row['label'] + '.json')); request = raw['request']; events = raw['events']
        assert request['stream'] and len(request['input_ids']) == row['input_tokens'] == 1630
        assert request['sampling_params'] == {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True}
        assert request.get('lora_path') == ('fill-A' if side in ('nocache', 'lru', 'active') else None)
        previous = []; first = None; last_increase = None; clock = -1
        for event in events:
            ids = event['data'].get('output_ids', []); now = event['elapsed_s']
            assert math.isfinite(now) and now >= clock and ids[:len(previous)] == previous
            if len(ids) > len(previous):
                if first is None: first = event
                last_increase = now
            previous = ids; clock = now
        assert first is not None and len(previous) == row['output_tokens'] == 16
        assert previous == row['output_ids'] and 0 < row['first_event_tokens'] == len(first['data']['output_ids']) < 16
        equal(first['elapsed_s'], row['ttft_s'])
        equal((last_increase - first['elapsed_s']) / (16 - row['first_event_tokens']), row['tpot_s'])
        assert math.isfinite(row['e2e_s']) and row['e2e_s'] >= events[-1]['elapsed_s']
        meta = events[-1]['data']['meta_info']
        assert meta['prompt_tokens'] == 1630 and meta['completion_tokens'] == 16
        assert meta.get('cached_tokens', 0) == row['cached_tokens'] == 0
        assert meta['finish_reason'] == {'type': 'length', 'length': 16}
    api = read(directory / 'api.json')
    assert sum(v['path'] == '/flush_cache' for v in api) == 10
    for route in ('/load_lora_adapter', '/unload_lora_adapter'):
        assert sum(v['path'] == route for v in api) == (5 if side in ('nocache', 'lru', 'active') else 0)
    assert all(math.isfinite(v['elapsed_s']) and v['elapsed_s'] > 0 for v in api)
    assert all(r['output_ids'] == rows[0]['output_ids'] for r in rows)
    assert len(summary['points']) == 2
    for point in summary['points']:
        selected = [r for r in rows if r['mode'] == point['mode']]; assert len(selected) == point['requests'] == 5
        for metric in ('ttft_s', 'tpot_s', 'e2e_s'):
            values = sorted(r[metric] for r in selected)
            for name, value in {'mean': math.fsum(values) / 5, **{f'p{int(q*100)}': values[math.ceil(5*q)-1] for q in (.5, .95, .99)}}.items(): equal(value, point[metric][name])
    return {'summary': summary, 'output_ids': rows[0]['output_ids']}


def main():
    source = read(R / 'source-host-r147.json')
    for name, digest in source['source_files'].items(): assert sha(R / name) == digest, name
    output = R / 'active-isolated-audit-r147.json'; assert not output.exists()
    folders = [R / f'perf-active-isolated-{i}-{s}-r147' for i, s in enumerate(ORDER)]
    folders += [R / f'fill-active-isolated-{i}-{s}-r147' for i, s in enumerate(ORDER[:5])]
    inventories = {p.name: inventory(p) for p in folders}
    placements = lines(R / 'actual-placement-observations-r147.jsonl'); assert len(placements) == 8
    controls = [native(i, s, placements) for i, s in enumerate(ORDER)]
    reference = controls[3]['records']; common = ('tp_size','attn_dp_size','pp_size','dtype','kv_cache_dtype','context_length','chunked_prefill_size','max_running_requests','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','linear_attn_backend','linear_attn_prefill_backend','linear_attn_decode_backend','bf16_gemm_backend','enable_deterministic_inference','disable_cuda_graph','mamba_ssm_dtype','mamba_radix_cache_strategy','page_size')
    intervals = []
    for i, control in enumerate(controls):
        assert all(control['configuration'].get(k) == controls[3]['configuration'].get(k) for k in common)
        assert set(control['records']) == set(reference)
        for label, row in control['records'].items():
            assert row['input_ids'] == reference[label]['input_ids'] and row['sampling_params'] == reference[label]['sampling_params']
            if ORDER[i] != 'base': assert row['output_ids'] == reference[label]['output_ids']
        control.pop('records')
    monitor = lines(R / 'active-isolated-observations-r147.jsonl')
    assert len(monitor) >= 10 and not any(v.get('unexpected') or v.get('error') for v in monitor)
    assert all(len(v['rows']) == 8 and all(int(g[1]) < 2000 and int(g[2]) == 0 for g in v['rows'] if int(g[0]) != 2) for v in monitor)
    for p in folders:
        for row in lines(p / 'requests.jsonl'):
            end = (p / (row['label'] + '.json')).stat().st_mtime
            intervals.append((end - row['e2e_s'], end))
    assert monitor[0]['unix_s'] <= min(a for a, b in intervals) and monitor[-1]['unix_s'] >= max(b for a, b in intervals) - 2
    fills = [fill(i, s) for i, s in enumerate(ORDER[:5])]
    assert fills[0]['output_ids'] == fills[1]['output_ids'] == fills[2]['output_ids'] == fills[3]['output_ids']
    points = []
    for indexes in ((0, 1, 2), (7, 6, 5)):
        for point in range(16):
            rates = {ORDER[i]: controls[i]['summary']['points'][point]['throughput_tokens_s_mean'] for i in indexes}
            key = {k: controls[3]['summary']['points'][point][k] for k in ('input_tokens','concurrency','cache')}
            assert all(all(controls[i]['summary']['points'][point][k] == v for k, v in key.items()) for i in indexes)
            points.append({**key, 'positions': indexes, 'rates': rates, 'active_over_lru': rates['active']/rates['lru'], 'active_over_nocache': rates['active']/rates['nocache']})
    assert all(inventory(p) == inventories[p.name] for p in folders)
    report = {'passed': True, 'full_goal_complete': False, 'scope': source['scope'],
              'natural_requests': 5088, 'natural_latency_statistics_recomputed': 1536,
              'fill_requests': 50, 'fill_latency_statistics_recomputed': 120,
              'controls': controls, 'fill': fills, 'speed_points': points,
              'monitor_samples': len(monitor), 'source_manifest_sha256': sha(R/'source-host-r147.json'),
              'raw_files_sha256': inventories, 'auditor_sha256': sha(Path(__file__)),
              'original_native_auditor_sha256': sha(R/'audit_topology_native_r100.py'),
              'audit_transformation':'One original native loop arm; TP1/memory.52/no-profile; remove only profiler section, retain SSE/numerical-time/label/group/percentile checks. Explicit actual placement/image/model/config gates added.'}
    with output.open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ('passed','natural_requests','fill_requests','monitor_samples')}))


if __name__ == '__main__': main()
