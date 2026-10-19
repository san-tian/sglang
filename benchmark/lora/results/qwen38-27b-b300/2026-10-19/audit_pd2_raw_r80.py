"""Read-only reconstruction of real 2P2D scores and actual rank placement."""
import copy
import json
from pathlib import Path
from audit_active_raw_r65 import read, digest, inventory, numerical_stats, check_statistics

ROOT = Path(__file__).resolve().parent
PHASES = ('prefill', 'prefill_last', 'decode')


def placement(label):
    proof = read(ROOT / f'pd2-placement-{label}-r80.json')
    gpu_by_uuid = {row.split(',')[1].strip(): int(row.split(',')[0])
                   for row in proof['gpu_inventory'].splitlines()}
    gpu_by_pid = {}
    for row in proof['gpu_processes'].splitlines():
        pid, uuid, _ = row.split(',')
        gpu_by_pid.setdefault(int(pid), set()).add(gpu_by_uuid[uuid.strip()])
    result = {}
    for name, item in proof['containers'].items():
        pids = {int(row.split()[0]) for row in item['host_processes'].splitlines()[1:]}
        actual = set().union(*(gpu_by_pid.get(pid, set()) for pid in pids))
        expected = {0, 3} if '-prefill-' in name else {4, 5}
        assert actual == expected, (name, actual, expected)
        assert item['pid_mode'] == 'host' and item['state']['Running']
        assert item['image'] == read(ROOT / 'source-host-r80.json')['runtime_image']
        assert item['devices'][0]['DeviceIDs'] == ['0', '3', '4', '5']
        result[name] = sorted(actual)
    assert len(result) == 2
    return result


def original_scores(folder, name, case_index, case, continuation, merged):
    # Locate original P/D responses by the capture path, without trusting the
    # client's combined response or intermediate score files.
    p_request = p_response = d_request = d_response = None
    for pfile in folder.glob('pd-*-prefill.json'):
        prefill = read(pfile)
        if 'custom_logit_processor' not in prefill['requests'][0]:
            continue
        req = prefill['requests'][case_index]
        capture = req['sampling_params']['custom_params']['capture_file']
        if Path(capture).name != f'{name}-i{case_index}.jsonl.prefill':
            continue
        assert p_request is None, 'ambiguous original transfer'
        decode = read(pfile.with_name(pfile.name.replace('-prefill.json', '-decode.json')))
        p_request, p_response = req, prefill['responses'][case_index]
        d_request, d_response = decode['requests'][case_index], decode['responses'][case_index]
    assert p_request is not None
    assert p_request['bootstrap_room'] == d_request['bootstrap_room']
    assert p_request['input_ids'] == d_request['input_ids'] == case['input_ids']
    for role, request, expected_step in [('prefill', p_request, 0), ('decode', d_request, 1)]:
        params = request['sampling_params']['custom_params']
        assert params['continuation'] == continuation and params['capture_start_step'] == expected_step
        assert Path(params['capture_file']).name == f'{name}-i{case_index}.jsonl.{role}'
    assert p_response['output_ids'] == continuation[:1]
    assert d_response['output_ids'] == continuation
    combined = copy.deepcopy(d_response)
    for key, value in p_response['meta_info'].items():
        if key.startswith('input_'):
            combined['meta_info'][key] = value
    assert combined == merged
    rows = []
    for role, steps in [('prefill', [0]), ('decode', list(range(1, 16)))]:
        paths = sorted((folder / 'captures').glob(f'{name}-i{case_index}.jsonl.{role}.rank-*'))
        assert len(paths) == 2
        assert {p.name.rsplit('-', 1)[1] for p in paths} == {'0', '1'}
        per_rank = [[json.loads(x) for x in p.read_text().splitlines()] for p in paths]
        assert per_rank[0] == per_rank[1]
        assert [r['step'] for r in per_rank[0]] == steps
        assert [r['token'] for r in per_rank[0]] == [continuation[s] for s in steps]
        rows.extend(per_rank[0])
    meta = p_response['meta_info']
    assert meta['prompt_tokens'] == len(case['input_ids'])
    lp, top = meta['input_token_logprobs'], meta['input_top_logprobs']
    assert len(lp) == len(top) == len(case['input_ids'])
    assert [x[1] for x in lp[1:]] == case['input_ids'][1:]
    return {'prefill': [(p[0], t[0][1]) for p, t in zip(lp[1:], top[1:])],
            'prefill_last': [(rows[0]['logprob'], rows[0]['argmax'])],
            'decode': [(r['logprob'], r['argmax']) for r in rows[1:]]}


def audit(label):
    folders = {side: ROOT / f'pd2diag-{label}-{stage}-r80'
               for side, stage in [('M', 'reference'), ('D', 'dynamic')]}
    snapshots = {s: inventory(f) for s, f in folders.items()}
    seal = read(folders['M'] / 'seal.json')['files']
    assert snapshots['M'] == dict(seal, **{'seal.json': digest(folders['M'] / 'seal.json')})
    contracts = {s: read(f / 'contract.json') for s, f in folders.items()}
    assert contracts['D'] == contracts['M']
    contract = contracts['M']
    assert (contract['topology'], contract['rounds'], contract['decode_tokens'],
            contract['batch_size']) == ('pd2diag-r80', 1, 16, 4)
    corpus = contract['corpus']
    assert len(corpus) == len({c['id'] for c in corpus}) == 4
    for role, filename, gpu_base in [('prefill', 'prefill-server-info.json', 0),
                                     ('decode', 'server-info.json', 2)]:
        configs = {}
        for side, folder in folders.items():
            cfg = read(folder / filename)
            cfg = cfg.get('server_args', cfg)
            configs[side] = cfg
            assert cfg['tp_size'] == 2 and cfg['attn_dp_size'] == cfg['pp_size'] == 1
            assert cfg['base_gpu_id'] == gpu_base and cfg['disaggregation_mode'] == role
            assert cfg['disaggregation_transfer_backend'] == 'nixl'
            assert cfg['disaggregation_enable_kv_checksum']
            assert cfg['mamba_radix_cache_strategy'] == 'extra_buffer'
            assert cfg['quantization'] is None and not cfg['enable_tf32_matmul']
            assert cfg['enable_deterministic_inference'] and cfg['disable_cuda_graph']
            assert bool(cfg['enable_lora']) == (side == 'D')
            assert read(folder / 'provenance.json') == read(ROOT / 'source-host-r80.json')
        keys = ('dtype', 'context_length', 'chunked_prefill_size', 'max_running_requests',
                'attention_backend', 'linear_attn_backend', 'linear_attn_prefill_backend',
                'linear_attn_decode_backend', 'bf16_gemm_backend', 'kv_cache_dtype',
                'max_mamba_cache_size', 'json_model_override_args')
        assert all(configs['M'].get(k) == configs['D'].get(k) for k in keys)
    scores = {phase: [] for phase in PHASES}
    noise = {side: {phase: [] for phase in PHASES} for side in folders}
    first = {}
    counts = {side: 0 for side in folders}
    for cache in ('cold', 'hot'):
        name = 'g0-r0-' + cache
        responses = {s: read(f / f'{name}-responses.json') for s, f in folders.items()}
        refs = read(folders['M'] / 'references-0.json')
        assert read(folders['D'] / 'references-0.json') == refs
        per_case = {s: [json.loads(x) for x in (f / 'cases.jsonl').read_text().splitlines()]
                    for s, f in folders.items()}
        for i, case in enumerate(corpus):
            continuation = refs[i]['output_ids']
            assert len(continuation) == 16
            actual = {s: original_scores(f, name, i, case, continuation, responses[s][i])
                      for s, f in folders.items()}
            for side in folders:
                row = per_case[side][counts[side]]
                counts[side] += 1
                assert (row['case'], row['round'], row['cache']) == (case['id'], 0, cache)
                for phase in PHASES:
                    reference = actual['M'][phase] if side == 'D' else actual[side][phase]
                    assert len(actual[side][phase]) == len(reference)
                    check_statistics(numerical_stats(list(zip(actual[side][phase], reference))), row[phase])
                    key = side, i, phase
                    if key in first:
                        noise[side][phase].extend(zip(first[key], actual[side][phase]))
                    else:
                        first[key] = actual[side][phase]
            for phase in PHASES:
                scores[phase].extend(zip(actual['D'][phase], actual['M'][phase]))
    recomputed = {p: numerical_stats(v) for p, v in scores.items()}
    summaries = {s: read(f / 'summary.json') for s, f in folders.items()}
    assert counts == {'M': 8, 'D': 8}
    for side in folders:
        assert summaries[side]['case_pairs'] == 8
        for phase in PHASES:
            check_statistics(numerical_stats(noise[side][phase]), summaries[side]['repeat_noise'][phase])
        assert inventory(folders[side]) == snapshots[side], 'raw inputs changed'
    for phase in PHASES:
        check_statistics(recomputed[phase], summaries['D']['scores'][phase])
    return {'passed': True, 'case_pairs': 8, 'scores': recomputed,
            'raw_files_rehashed': {s: len(v) for s, v in snapshots.items()},
            'rank_capture_files_reparsed': 64,
            'reference_seal_sha256': digest(folders['M'] / 'seal.json')}


if __name__ == '__main__':
    output = ROOT / 'pd2-raw-accuracy-audit-r80.json'
    assert not output.exists()
    assert read(ROOT / 'pd2-diagnostic-complete-r80.json')['natural_output_ids_equal']
    result = {'passed': True,
              'scope': 'Additional real 2P2D diagnostic only; full 4P4D and speed acceptance remain pending.',
              'placement': {label: placement(label) for label in ('m-real_r16', 'm-gdn_fused', 'd')},
              'precision': {label: audit(label) for label in ('full', 'gdn')},
              'auditor_sha256': digest(Path(__file__))}
    result['score_count'] = sum(v['count'] for a in result['precision'].values() for v in a['scores'].values())
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
