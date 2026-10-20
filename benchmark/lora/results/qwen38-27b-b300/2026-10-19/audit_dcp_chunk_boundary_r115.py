"""Correlate frozen scheduler logs and raw prefill errors; not a causal proof."""
import hashlib
import json
import re
from pathlib import Path

from audit_cross_topology_prefill_r113 import prefill_scores

R = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    source = read(R / 'source-dcp-boundary-r115.json')
    for name, digest in source['source_files'].items():
        assert sha(R / name) == digest, name
    original = read(R / 'tp8-vs-dcp8x2-merged-prefill-r113.json')
    assert original['passed'] is False and original['case_pairs'] == 240
    directories = {top: R / f'{top}-full-reference-r30' for top in ('tp8', 'dcp8x2')}
    contracts = {top: read(d / 'contract.json') for top, d in directories.items()}
    corpus = contracts['tp8']['corpus']
    assert corpus == contracts['dcp8x2']['corpus'] and len(corpus) == 30
    assert sum(len(c['input_ids']) for c in corpus) == 6248
    for top, d in directories.items():
        assert sha(d / 'seal.json') == original['reference_seals'][top]['sha256']

    # Infer a FIFO decomposition, then check both observed sequence counts
    # and token counts. Logs do not expose per-request IDs: retain this limit.
    token_counts, sequence_counts = [2008, 2000, 2018, 222], [27, 3, 2, 1]
    index, offset, cuts, allocation = 0, 0, {}, []
    for budget, expected_seqs in zip(token_counts, sequence_counts):
        batch = []
        while budget:
            case = corpus[index]
            length = len(case['input_ids'])
            used = min(budget, length - offset)
            batch.append({'case': case['id'], 'start': offset, 'end': offset + used})
            offset += used
            budget -= used
            if offset == length:
                index += 1
                offset = 0
            else:
                cuts.setdefault(case['id'], []).append(offset)
        assert len(batch) == expected_seqs
        allocation.append(batch)
    assert index == 30 and offset == 0

    logs = {}
    for top, name in source['scheduler_logs'].items():
        p = R / name
        assert sha(p) == source['scheduler_log_sha256'][top]
        prefill = []
        for number, text in enumerate(p.read_text().splitlines(), 1):
            m = re.search(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)', text)
            if m:
                prefill.append({'line': number, 'seqs': int(m[1]), 'tokens': int(m[2]),
                                'cached': int(m[3]), 'text': text})
        expected = list(zip(sequence_counts, token_counts))
        matches = [prefill[i:i+4] for i in range(len(prefill)-3)
                   if [(r['seqs'], r['tokens']) for r in prefill[i:i+4]] == expected]
        assert len(matches) == 8 and all(r['cached'] == 0 for batch in matches for r in batch)
        logs[top] = {'file': name, 'sha256': sha(p), 'matching_four_batch_groups': matches}

    observations = []
    for repeat in range(4):
        for cache in ('cold', 'hot'):
            name = f'g0-r{repeat}-{cache}-responses.json'
            values = {}
            for top, d in directories.items():
                assert sha(d / name) == original['raw_response_sha256'][name][top]
                values[top] = read(d / name)
                assert len(values[top]) == 30
            for i, case in enumerate(corpus):
                a, b = [prefill_scores(values[top][i], case) for top in ('tp8', 'dcp8x2')]
                errors = [(position, abs(x[0]-y[0]))
                          for position, (x, y) in enumerate(zip(a, b), 1)]
                nonzero = [position for position, error in errors if error > 0]
                above = [position for position, error in errors if error > .1]
                if not nonzero:
                    assert case['id'] not in cuts
                    continue
                assert above and case['id'] in cuts and len(cuts[case['id']]) == 1
                cut = cuts[case['id']][0]
                assert nonzero[0] == above[0] == cut + 1
                observations.append({'case': case['id'], 'round': repeat, 'cache': cache,
                                     'input_tokens': len(case['input_ids']),
                                     'inferred_prefix_length': cut,
                                     'first_nonzero_input_index_0based': nonzero[0],
                                     'first_above_0p1_input_index_0based': above[0]})
    assert len(observations) == 24
    return {'correlation_verified': True, 'root_cause_proven': False,
            'full_goal_complete': False,
            'scope': 'Frozen scheduler logs and raw input logprobs correlate the first cross-topology difference with the first scored token after a prefix/current split. FIFO per-request decomposition is inferred from corpus order and matching aggregate sequence/token counts; logs lack request IDs. No intervention or per-layer capture proves causality. No new GPU experiment or acceptance.',
            'source_manifest_sha256': sha(R / 'source-dcp-boundary-r115.json'),
            'prefill_batch_tokens': token_counts, 'prefill_batch_sequence_counts': sequence_counts,
            'fifo_inferred_allocation': allocation, 'inferred_chunk_cuts': cuts,
            'scheduler_logs': logs, 'observations': observations}


if __name__ == '__main__':
    out = R / 'dcp-chunk-boundary-correlation-r115.json'
    assert not out.exists()
    result = audit()
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('correlation_verified', 'root_cause_proven', 'inferred_chunk_cuts')}))
