"""Audit completed DP8 D/M native controls using original R135 SSE/rank checks.

Only arm selection and independently generated profiler-summary reads change.
Producer directories, labels, time/finite/percentile and actual rank gates stay.
This accepts reconstructed measurements, not a speed improvement or KV capacity.
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


def audit_one(root, side):
    top = 'dpa8'
    dp = True
    assert side in ('d', 'm')
    filename = ('audit_attention_dp8_native_r135.py' if dp else
                'audit_topology_native_r100.py')
    manifest_name = ('source-attention-dp8-audits-r135.json' if dp else
                     'source-topology-audits-r102.json')
    source = read(root / manifest_name)
    assert sha(root / filename) == source['source_files'][filename]
    directory = root / f'perf-{top}-{side}-r135'
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
    loop.iter = ast.List(elts=[ast.Constant(value=side)], ctx=ast.Load())
    summary = root / f'dpa8-{side}-kernel-summary-r151.json'
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
    exec(compile(body, filename + ':single-arm-R151', 'exec'), ns)
    assert ns['latency_statistics'] == 192
    assert len(ns['rows_by_side'][side]) == 636 and len(ns['proofs'][side]) == 636
    traces = ns['traces_by_side'][side]
    assert len(traces) == 8 and {t['rank'] for t in traces} == set(range(8))
    if side == 'm':
        assert all(t['merge_round'] == 0 for t in traces)
    for trace in traces:
        trace.pop('all_exact_kernel_counts')
    return {'passed': True, 'native_requests': 636, 'latency_statistics_recomputed': 192,
            'request_payload_sha256': hashlib.sha256(json.dumps({
                label: {k: row[k] for k in ('input_ids', 'sampling_params')}
                for label, row in ns['rows_by_side'][side].items()
            }, sort_keys=True).encode()).hexdigest(),
            'natural_output_ids_sha256': hashlib.sha256(json.dumps({label: row['output_ids'] for label,row in ns['rows_by_side'][side].items()},sort_keys=True).encode()).hexdigest(),
            'configuration': ns['configurations'][side],
            'speed_points': ns['summaries'][side]['points'],
            'profile_traces': traces, 'placement': ns['placement_proofs'][side],
            'raw_native_sha256': ns['proofs'][side],
            'original_auditor_sha256': sha(root / filename),
            'original_auditor_manifest_sha256': sha(root / manifest_name),
            'transformed_auditor_ast_sha256': hashlib.sha256(ast.dump(body).encode()).hexdigest(),
            'transformations': [f'select {side} arm only'] + (
                ['read independently generated kernel summary at diagnostic path'] if dp else []),
            'profile_summary_sha256': sha(summary if dp else directory / 'kernel-summary.json')}



def main(root, output):
    assert not output.exists()
    source = read(root / 'source-dpa8-native-stage-r151.json')
    for name,digest in source['source_files'].items():
        assert sha(root/name) == digest,name
    results = {side: audit_one(root, side) for side in ('d','m')}
    d,m = results['d'],results['m']
    assert d['request_payload_sha256'] == m['request_payload_sha256']
    assert d['natural_output_ids_sha256'] == m['natural_output_ids_sha256']
    common=('tp_size','attn_dp_size','context_length','dtype','max_mamba_cache_size',
            'mem_fraction_static','attention_backend','page_size','chunked_prefill_size',
            'max_running_requests','lora_merged_bf16_cache_size_mb','dcp_size',
            'dcp_comm_backend','enable_deterministic_inference','disable_cuda_graph',
            'linear_attn_prefill_backend','bf16_gemm_backend','kv_cache_dtype')
    assert all(d['configuration'].get(k) == m['configuration'].get(k) for k in common)
    assert min(v['configuration']['max_total_num_tokens'] for v in results.values()) > 73728
    points=[]
    for a,b in zip(d['speed_points'],m['speed_points']):
        key=('input_tokens','concurrency','cache')
        assert all(a[k] == b[k] for k in key)
        points.append({**{k:a[k] for k in key},'D_tok_s':a['throughput_tokens_s_mean'],
                       'M_tok_s':b['throughput_tokens_s_mean'],
                       'D_over_M':a['throughput_tokens_s_mean']/b['throughput_tokens_s_mean'],
                       'D_e2e_p95_s':a['e2e_s']['p95'],'M_e2e_p95_s':b['e2e_s']['p95'],
                       'D_ttft_p95_s':a['ttft_s']['p95'],'M_ttft_p95_s':b['ttft_s']['p95'],
                       'D_tpot_p95_s':a['tpot_s']['p95'],'M_tpot_p95_s':b['tpot_s']['p95']})
    report={'passed':True,'full_goal_complete':False,'natural_requests':1272,
            'latency_statistics_recomputed':384,'rank_traces':16,'natural_output_ids_match':True,
            'controls':results,'speed_points':points,'source_manifest_sha256':sha(root/'source-dpa8-native-stage-r151.json'),
            'optimization_accepted':False,'KV_capacity_accepted':False,'base_control_accepted':False,
            'scope':'Original R135 R28-image DP8 same-topology dynamic no-cache MoL versus independent merge native controls only. Each636natural responses,192latency statistics and8actual rank traces; source, raw payloads, time/finite/percentile, actual cgroup/scheduler/GPU placements and input/natural ID equality checked. Auto KV rows may differ, both exceed73728requested tokens; not equal-byte capacity evidence. Cold/hot ratios are separate-time observations, not isolated or universally causal. Original mixed and precision stages passed separately; cross-topology R139 remains failed. No R55 optimization, base, resident-KV or full-goal acceptance.'}
    with output.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='controls'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
