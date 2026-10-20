"""Read-only TP8/DCP2 merged-reference prefill diagnostic, not D/M acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_topology_original_r99 import numerical_stats


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefill_scores(response, case):
    meta = response['meta_info']
    ids = case['input_ids']
    assert meta['prompt_tokens'] == len(ids)
    scores, tops = meta['input_token_logprobs'], meta['input_top_logprobs']
    assert len(scores) == len(tops) == len(ids)
    assert [row[1] for row in scores[1:]] == ids[1:]
    result = [(score[0], top[0][1]) for score, top in zip(scores[1:], tops[1:])]
    assert result and all(math.isfinite(v) and type(token) is int for v, token in result)
    return result


def audit(root):
    source = read(root/'source-topology-audits-r102.json')
    assert sha(root/'audit_topology_original_r99.py') == source['source_files']['audit_topology_original_r99.py']
    dirs = {top: root/f'{top}-full-reference-r30' for top in ('tp8', 'dcp8x2')}
    contracts, configs, seals = {}, {}, {}
    for top, directory in dirs.items():
        seals[top] = seal = read(directory/'seal.json')
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        assert actual == set(seal['files']) | {'seal.json'}
        assert all(sha(directory/name) == digest for name, digest in seal['files'].items())
        contracts[top] = read(directory/'contract.json')
        info = read(directory/'server-info.json')
        configs[top] = info.get('server_args', info)
        assert configs[top]['tp_size'] == 8 and configs[top]['attn_dp_size'] == 1
        assert configs[top]['dcp_size'] == (2 if top == 'dcp8x2' else 1)
        assert read(directory/'provenance.json') == read(root/'source-host-r85.json')
    a, b = (dict(contracts[top]) for top in dirs)
    assert a.pop('topology') == 'tp8-r30' and b.pop('topology') == 'dcp8x2-r30'
    assert a == b
    for key in ('model_path', 'dtype', 'kv_cache_dtype', 'context_length',
                'attention_backend', 'bf16_gemm_backend', 'enable_deterministic_inference',
                'disable_cuda_graph', 'enable_tf32_matmul', 'json_model_override_args',
                'max_mamba_cache_size', 'mamba_radix_cache_strategy'):
        assert configs['tp8'][key] == configs['dcp8x2'][key], key
    corpus = a['corpus']
    assert len(corpus) == 30 and a['rounds'] == 4 and a['batch_size'] == 30
    pairs, cases, response_hashes = [], [], {}
    for repeat in range(4):
        for cache in ('cold', 'hot'):
            name = f'g0-r{repeat}-{cache}-responses.json'
            responses = {top: read(directory/name) for top, directory in dirs.items()}
            assert all(len(v) == len(corpus) for v in responses.values())
            response_hashes[name] = {top: sha(directory/name) for top, directory in dirs.items()}
            for i, case in enumerate(corpus):
                left = prefill_scores(responses['tp8'][i], case)
                right = prefill_scores(responses['dcp8x2'][i], case)
                assert len(left) == len(right) == len(case['input_ids'])-1
                values = list(zip(left, right))
                pairs.extend(values)
                cases.append({'case': case['id'], 'round': repeat, 'cache': cache,
                              'prefill': numerical_stats(values)})
    aggregate = numerical_stats(pairs)
    assert len(cases) == 240 and aggregate['count'] == 49744
    passed = aggregate['passes_logprob_gate'] and all(c['prefill']['passes_logprob_gate'] for c in cases)
    return {'passed': passed, 'full_goal_complete': False,
            'scope': 'Cross-topology TP8 versus DCP2 merged-reference prefill only. Identical fixed input tokens; no decode scores on differing continuation paths are compared. This neither accepts dynamic MoL precision nor native speed, KV residency, or the overall goal.',
            'case_pairs': len(cases), 'prefill': aggregate, 'cases': cases,
            'reference_seals': {top: {'sha256': sha(directory/'seal.json'), 'files_rehashed': len(seals[top]['files'])} for top, directory in dirs.items()},
            'raw_response_sha256': response_hashes,
            'diagnostic_source_sha256': sha(Path(__file__)),
            'statistics_source_sha256': sha(root/'audit_topology_original_r99.py')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('passed', 'scope', 'case_pairs', 'prefill')}))
    raise SystemExit(0 if result['passed'] else 1)
