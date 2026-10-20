"""Independent raw audits for R55 TP4/DP4 cache controls.

Reuse original numerical/rank/SSE/percentile audits. A documented path map points
them at immutable new producer directories; no original experiment is changed.
"""
import argparse
import ast
import collections
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import audit_topology_original_r99 as precision
import audit_topology_mixed_r101 as mixed_audit

R = Path(__file__).resolve().parent
IMAGE = 'sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
COMMIT = '0116d3a12bc893ae43d8dbfd594280eccdc8467c'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Paths:
    """Explicit audit-input mapping; all operations return actual pathlib paths."""
    def __init__(self, mappings):
        self.mappings = mappings
        self.parent = R.parent

    def __truediv__(self, name):
        return self.mappings.get(name, R / name)


def accuracy(top, policy):
    mappings = {'source-host-r85.json': R / 'source-host-r144.json'}
    for label in ('full', 'gdn'):
        mappings[f'{top}-{label}-reference-r30'] = R / f'active-{top}-{label}-reference-r144'
        mappings[f'{top}-{label}-dynamic-r30'] = R / f'active-{top}-{policy}-{label}-dynamic-r144'
    tree = ast.parse((R / 'audit_topology_original_r99.py').read_text())
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit'))
    replaced = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Assert) and ast.unparse(node.test) == "cfg.get('lora_merged_bf16_cache_size_mb', 0) == 0":
            node.test = ast.parse("cfg.get('lora_merged_bf16_cache_size_mb', 0) == (budget if side == 'D' else 0)", mode='eval').body
            replaced += 1
    assert replaced == 1
    ns = dict(vars(precision))
    ns.update(IMAGE=IMAGE, COMMIT=COMMIT, budget=12288 if top == 'dpa4' else 10240)
    body = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(body, 'audit_topology_original_r99.py:R144-cache-budget', 'exec'), ns)
    reports = {}
    for label, n, repeats, tokens, counts in [
        ('full', 30, 4, 256, {'prefill': 49744, 'prefill_last': 240, 'decode': 61200}),
        ('gdn', 4, 1, 16, {'prefill': 3380, 'prefill_last': 8, 'decode': 120})]:
        cfg = read(mappings[f'{top}-{label}-dynamic-r30'] / 'server-info.json')
        cfg = cfg.get('server_args', cfg)
        assert cfg['lora_merged_bf16_cache_policy'] == policy
        assert cfg['max_total_num_tokens'] == 65536
        reports[label] = ns['audit'](Paths(mappings), top, label, n, repeats, tokens, counts, False)
    assert sum(v['case_pairs'] for v in reports.values()) == 248
    assert sum(s['count'] for v in reports.values() for s in v['scores'].values()) == 114692
    return {'accuracy': reports, 'original_auditor_sha256': sha(R / 'audit_topology_original_r99.py'),
            'audit_ast_sha256': hashlib.sha256(ast.dump(body).encode()).hexdigest(),
            'transformation': 'Only expected D cache budget changes from zero; original raw numerical checks unchanged.',
            'path_mapping': {k: str(v) for k, v in mappings.items()}}


def native(top, side, placements):
    label = side if side in ('m', 'base') else 'd'
    suffix = 'r41' if side == 'base' else 'r30'
    launch_alias = f'launch-qwen38-{top}-' + ('m-real_r16-r30' if side == 'm' else
                    'base-r41' if side == 'base' else 'd-r30') + '.json'
    actual_name = f'qwen38-active-{top}-' + ('m-real_r16' if side == 'm' else side) + '-r144'
    directory = R / f'perf-active-{top}-{side}-r144'
    mappings = {f'perf-{top}-{label}-{suffix}': directory,
                launch_alias: R / f'launch-{actual_name}.json'}
    tree = ast.parse((R / 'audit_topology_native_r100.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert {n.name for n in functions} == {'read', 'lines', 'sha', 'cfg'}
    loops = [n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == 'side' and ast.literal_eval(n.iter) == ['d', 'm', 'base']]
    assert len(loops) == 1
    loop = copy.deepcopy(loops[0])
    loop.iter = ast.List(elts=[ast.Constant(value=label)], ctx=ast.Load())
    ns = {k: globals()[k] for k in ('collections', 'gzip', 'hashlib', 'json', 'math', 'Path')}
    ns.update(R=Paths(mappings), top=top, legacy=True, tp=4,
              dpa=top == 'dpa4', dcp=1, IMAGE=IMAGE,
              expected_gpus='"device=0,1,2,3"', latency_statistics=0)
    for name in ('placement_proofs', 'configurations', 'rows_by_side', 'proofs',
                 'traces_by_side', 'summaries', 'launches'):
        ns[name] = {}
    body = ast.fix_missing_locations(ast.Module(body=functions + [loop], type_ignores=[]))
    exec(compile(body, 'audit_topology_native_r100.py:R144-control', 'exec'), ns)
    assert ns['latency_statistics'] == 192 and len(ns['rows_by_side'][label]) == 636
    cfg = ns['configurations'][label]
    assert cfg['max_total_num_tokens'] == 65536 and cfg['max_mamba_cache_size'] == 256
    assert cfg['lora_merged_bf16_cache_size_mb'] == ((12288 if top == 'dpa4' else 10240) if side in ('active', 'lru') else 0)
    if side in ('active', 'lru'):
        assert cfg['lora_merged_bf16_cache_policy'] == side
    command = read(mappings[launch_alias])
    assert IMAGE in command
    assert command[command.index('--model-path') + 1] == (
        '/work/qwen38-mol-development-20261019/merged/real_r16' if side == 'm' else
        '/work/models/Qwen3.8-27B/' if side == 'base' else '/work/models/Qwen3.8-27B')
    matches = [v for v in placements if v['name'] == actual_name]
    assert len(matches) == 1
    placement = matches[0]
    assert placement['placement_matched'] and placement['image'] == IMAGE
    assert placement['expected_compute_gpus'] == placement['observed_compute_gpus'] == list(range(4))
    assert len({v['pid'] for v in placement['scheduler_gpu_processes']}) == 4
    assert placement['command'] == command[command.index(IMAGE) + 1:]
    traces = ns['traces_by_side'][label]
    assert len(traces) == 4 and {v['rank'] for v in traces} == set(range(4))
    for trace in traces:
        trace.pop('all_exact_kernel_counts')
    return {'configuration': cfg, 'speed_points': ns['summaries'][label]['points'],
            'raw_native_sha256': ns['proofs'][label], 'profile_traces': traces,
            'actual_placement': placement, 'native_requests': 636,
            'latency_statistics_recomputed': 192,
            'audit_ast_sha256': hashlib.sha256(ast.dump(body).encode()).hexdigest(),
            'original_auditor_sha256': sha(R / 'audit_topology_native_r100.py'),
            'path_mapping': {k: str(v) for k, v in mappings.items()},
            '_records': ns['rows_by_side'][label]}


def audit(top):
    source = read(R / 'source-host-r144.json')
    assert source['image'] == IMAGE and source['commit'] == COMMIT
    for name, digest in source['source_files'].items():
        assert sha(R / name) == digest, name
    output = R / f'active-{top}-full-audit-r144.json'
    assert not output.exists()
    placements = [json.loads(s) for s in (R / 'actual-placement-observations-r144.jsonl').read_text().splitlines()]
    policies = {}
    for policy in ('lru', 'active'):
        result = accuracy(top, policy)
        mixed_path = R / f'active-{top}-{policy}-mixed-raw-audit-r144.json'
        paths = Paths({f'{top}-mixed-r30': R / f'active-{top}-{policy}-mixed-r144',
                       f'{top}-mixed-captures-r30': R / f'active-{top}-{policy}-mixed-captures-r144'})
        mixed_audit.main(paths, mixed_path, top, True)
        result['mixed'] = read(mixed_path)
        assert result['mixed']['passed'] and result['mixed']['mixed_requests'] == 48
        policies[policy] = result
    controls = {side: native(top, side, placements) for side in ('m', 'base', 'nocache', 'lru', 'active')}
    reference = controls['m']['_records']
    for side, result in controls.items():
        assert set(result['_records']) == set(reference)
        for label, row in result['_records'].items():
            assert row['input_ids'] == reference[label]['input_ids']
            assert row['sampling_params'] == reference[label]['sampling_params']
            if side != 'base':
                assert row['output_ids'] == reference[label]['output_ids'], (side, label)
        result.pop('_records')
    observations_path = R / f'active-{top}-observations-r144.json'
    observations = read(observations_path)
    assert observations and not any(v.get('unexpected') or v.get('error') for v in observations)
    assert all(len(v['rows']) == 8 and all(int(row[1]) < 2000 and int(row[2]) == 0
               for row in v['rows'] if int(row[0]) >= 4) for v in observations)
    intervals = []
    for side in controls:
        folder = R / f'perf-active-{top}-{side}-r144'
        for row in [json.loads(s) for s in (folder / 'requests.jsonl').read_text().splitlines()]:
            end = (folder / (row['label'] + '.json')).stat().st_mtime
            intervals.append((end - row['e2e_s'], end))
    assert observations[0]['unix_s'] <= min(a for a, b in intervals)
    assert observations[-1]['unix_s'] >= max(b for a, b in intervals) - 2
    common = ('tp_size', 'attn_dp_size', 'context_length', 'dtype', 'max_total_num_tokens',
              'max_mamba_cache_size', 'mem_fraction_static', 'attention_backend', 'page_size',
              'chunked_prefill_size', 'max_running_requests', 'enable_deterministic_inference',
              'disable_cuda_graph', 'bf16_gemm_backend', 'kv_cache_dtype', 'mamba_radix_cache_strategy')
    for result in controls.values():
        assert all(result['configuration'].get(k) == controls['m']['configuration'].get(k) for k in common)
    points = []
    for i, m in enumerate(controls['m']['speed_points']):
        key = ('input_tokens', 'concurrency', 'cache')
        arms = {side: result['speed_points'][i] for side, result in controls.items()}
        assert all(all(p[k] == m[k] for k in key) for p in arms.values())
        rates = {side: p['throughput_tokens_s_mean'] for side, p in arms.items()}
        points.append({**{k: m[k] for k in key}, 'throughput_tok_s': rates,
                       'active_over_lru': rates['active'] / rates['lru'],
                       'active_over_nocache': rates['active'] / rates['nocache'],
                       'active_over_M': rates['active'] / rates['m']})
    report = {'passed': True, 'full_goal_complete': False, 'topology': top,
              'scope': source['scope'], 'runtime_image': IMAGE, 'precision_pairs': 496,
              'precision_scores': 229384, 'mixed_requests': 96, 'mixed_scores': 1536,
              'native_requests': 3180, 'latency_statistics_recomputed': 960,
              'actual_rank_traces': 20, 'policies': policies, 'controls': controls,
              'speed_points': points, 'gpu_observation_samples': len(observations),
              'gpu_observation_sha256': sha(observations_path),
              'source_manifest_sha256': sha(R / 'source-host-r144.json'),
              'auditor_sha256': sha(Path(__file__))}
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('policies', 'controls')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', choices=['tp4', 'dpa4'], required=True)
    audit(parser.parse_args().topology)
