"""Audit completed M-only baselines using unchanged original native-audit bodies.

The AST transformation selects only the merged arm and, for R135, redirects
the profiler summary read to a new diagnostic file. It does not edit producer
outputs, numerical gates, latency reconstruction, raw SSE or rank checks.
"""
import argparse
import ast
import collections
import copy
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_one(root, top):
    dp = top == 'dpa8'
    filename = ('audit_attention_dp8_native_r135.py' if dp else
                'audit_topology_native_r100.py')
    manifest_name = ('source-attention-dp8-audits-r135.json' if dp else
                     'source-topology-audits-r102.json')
    source = read(root / manifest_name)
    assert sha(root / filename) == source['source_files'][filename]
    directory = root / f'perf-{top}-m-{"r135" if dp else "r30"}'
    complete = read(directory / 'complete.json')
    assert complete == {'requests': 636, 'profile': True}
    native_source = ast.parse((root / filename).read_text())
    functions = [n for n in native_source.body if isinstance(n, ast.FunctionDef)]
    assert {n.name for n in functions} == {'read', 'lines', 'sha', 'cfg'}
    loops = [n for n in native_source.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'side'
             and ast.literal_eval(n.iter) == ['d', 'm', 'base']]
    assert len(loops) == 1
    original = loops[0]
    loop = copy.deepcopy(original)
    loop.iter = ast.List(elts=[ast.Constant(value='m')], ctx=ast.Load())
    summary = root / 'dpa8-merge-kernel-summary-r142.json'
    if dp:
        assert not summary.exists()
        subprocess.run(['python3', str(root / 'analyze_profiles_r30.py'),
                        str(directory / 'profile'), '--output', str(summary)], check=True)
        replaced = 0
        for node in ast.walk(loop):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'summarized' for t in node.targets):
                assert ast.unparse(node.value) == "read(directory / 'kernel-summary.json')"
                node.value = ast.parse('read(diagnostic_summary)', mode='eval').body
                replaced += 1
        assert replaced == 1
    ns = {key: globals()[key] for key in ('collections', 'gzip', 'hashlib', 'json', 'math', 'Path')}
    ns.update(R=root, top=top, legacy=False, tp=8, dpa=dp, dcp=1,
              IMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288',
              expected_gpus='"device=0,1,2,3,4,5,6,7"',
              placements=[json.loads(s) for s in (root / (
                  'actual-placement-observations-r135.jsonl' if dp else
                  'actual-placement-observations-r92.jsonl')).read_text().splitlines()],
              latency_statistics=0, diagnostic_summary=summary)
    for name in ('placement_proofs', 'configurations', 'rows_by_side', 'proofs',
                 'traces_by_side', 'summaries', 'launches'):
        ns[name] = {}
    body = ast.fix_missing_locations(ast.Module(body=functions + [loop], type_ignores=[]))
    exec(compile(body, filename + ':merged-only', 'exec'), ns)
    assert ns['latency_statistics'] == 192
    assert len(ns['rows_by_side']['m']) == 636 and len(ns['proofs']['m']) == 636
    traces = ns['traces_by_side']['m']
    assert len(traces) == 8 and {t['rank'] for t in traces} == set(range(8))
    assert all(t['merge_round'] == 0 for t in traces)
    for trace in traces:
        trace.pop('all_exact_kernel_counts')
    return {'passed': True, 'native_requests': 636, 'latency_statistics_recomputed': 192,
            'request_payload_sha256': hashlib.sha256(json.dumps({
                label: {k: row[k] for k in ('input_ids', 'sampling_params')}
                for label, row in ns['rows_by_side']['m'].items()
            }, sort_keys=True).encode()).hexdigest(),
            'configuration': ns['configurations']['m'],
            'speed_points': ns['summaries']['m']['points'],
            'profile_traces': traces, 'placement': ns['placement_proofs']['m'],
            'raw_native_sha256': ns['proofs']['m'],
            'original_auditor_sha256': sha(root / filename),
            'original_auditor_manifest_sha256': sha(root / manifest_name),
            'transformed_auditor_ast_sha256': hashlib.sha256(ast.dump(body).encode()).hexdigest(),
            'transformations': ['select merged arm only'] + (
                ['read independently generated kernel summary at diagnostic path'] if dp else []),
            'profile_summary_sha256': sha(summary if dp else directory / 'kernel-summary.json')}


def main(root, output):
    assert not output.exists()
    results = {top: audit_one(root, top) for top in ('tp8', 'dpa8')}
    cross = read(root / 'tp8-vs-attention-dp8-prefill-r139.json')
    assert cross['passed'] is False
    left, right = (results[t] for t in ('tp8', 'dpa8'))
    assert left['request_payload_sha256'] == right['request_payload_sha256']
    for key in ('model_path', 'dtype', 'context_length', 'max_mamba_cache_size',
                'enable_deterministic_inference', 'disable_cuda_graph', 'attention_backend'):
        assert left['configuration'][key] == right['configuration'][key], key
    points = []
    for a, b in zip(left['speed_points'], right['speed_points']):
        key = ('input_tokens', 'concurrency', 'cache')
        assert all(a[k] == b[k] for k in key)
        points.append({**{k: a[k] for k in key},
                       'tp8_M_tok_s': a['throughput_tokens_s_mean'],
                       'dpa8_M_tok_s': b['throughput_tokens_s_mean'],
                       'dpa8_over_tp8': b['throughput_tokens_s_mean'] / a['throughput_tokens_s_mean'],
                       'tp8_M_e2e_p95_s': a['e2e_s']['p95'],
                       'dpa8_M_e2e_p95_s': b['e2e_s']['p95']})
    report = {'passed': True, 'full_goal_complete': False,
              'scope': 'Completed independent-merge natural M controls only. Both use the '
                       'same original R28 image and physical GPUs0..7 at separate times. '
                       'The original raw SSE, 384 latency statistics and 16 rank traces are '
                       'rechecked. Cross-topology numerical parity remains FAILED. '
                       'These throughput ratios are observations and do not establish '
                       'MoL speed, an accepted optimization or equal-byte KV capacity.',
              'cross_topology_precision_accepted': False,
              'cross_topology_precision_report_sha256': sha(root / 'tp8-vs-attention-dp8-prefill-r139.json'),
              'controls': results, 'speed_points': points,
              'source_sha256': sha(Path(__file__))}
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'passed': True, 'native_requests': 1272,
                      'latency_statistics_recomputed': 384, 'rank_traces': 16,
                      'speed_points': points}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
