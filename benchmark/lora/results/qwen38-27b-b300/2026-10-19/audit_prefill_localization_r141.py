"""CPU-only localization of sealed TP8/attention-DP8 prefill discrepancies."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_topology_original_r99 import numerical_stats
from audit_tp8_vs_attention_dp8_r139 import audit, prefill_scores


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    # Recheck every source seal, contract, configuration and original score gate.
    cross = audit(root)
    recorded = read(root / 'tp8-vs-attention-dp8-prefill-r139.json')
    assert cross['prefill'] == recorded['prefill']
    assert cross['passed'] is recorded['passed'] is False
    dirs = {'tp8': root / 'tp8-full-reference-r30',
            'dpa8': root / 'dpa8-full-reference-r135'}
    corpus = read(dirs['tp8'] / 'contract.json')['corpus']
    buckets = {}
    within = {name: [] for name in dirs}
    baseline = {}
    cases = []
    positions = {}
    for repeat in range(4):
        for cache in ('cold', 'hot'):
            name = f'g0-r{repeat}-{cache}-responses.json'
            responses = {top: read(directory / name) for top, directory in dirs.items()}
            for i, case in enumerate(corpus):
                n = len(case['input_ids'])
                scores = {top: prefill_scores(rows[i], case) for top, rows in responses.items()}
                pairs = list(zip(scores['tp8'], scores['dpa8']))
                assert len(pairs) == n - 1
                bad = []
                for position, pair in enumerate(pairs, 1):
                    error = abs(pair[0][0] - pair[1][0])
                    if error > .1:
                        bad.append(position)
                    length = 'input_le256' if n <= 256 else 'input_gt256'
                    span = ('positions_1_127' if position < 128 else
                            'positions_128_255' if position < 256 else
                            'positions_256_511' if position < 512 else
                            'positions_ge512')
                    for key in (length, span, f'{length}/{span}'):
                        buckets.setdefault(key, []).append(pair)
                    stat = positions.setdefault(position, {'count': 0, 'abs_sum': 0.,
                                                           'max_abs_nat': 0., 'above_0_1': 0})
                    stat['count'] += 1
                    stat['abs_sum'] += error
                    stat['max_abs_nat'] = max(stat['max_abs_nat'], error)
                    stat['above_0_1'] += error > .1
                if repeat == 0 and cache == 'cold':
                    cases.append({'case': case['id'], 'input_tokens': n,
                                  'first_error_gt_0_1_position': min(bad) if bad else None,
                                  'positions_error_gt_0_1': len(bad),
                                  'prefill': numerical_stats(pairs)})
                    baseline[case['id']] = scores
                else:
                    for top in dirs:
                        within[top].extend(zip(baseline[case['id']][top], scores[top]))
    noise = {top: numerical_stats(pairs) for top, pairs in within.items()}
    assert all(s['max_abs_nat'] == 0 and s['argmax_match_fraction'] == 1 for s in noise.values())
    groups = {name: numerical_stats(pairs) for name, pairs in buckets.items()}
    for stat in positions.values():
        stat['mean_abs_nat'] = stat.pop('abs_sum') / stat['count']
    early = [c for c in cases if c['input_tokens'] <= 256 and
             c['first_error_gt_0_1_position'] is not None]
    return {'diagnosis_completed': True, 'accuracy_accepted': False,
            'full_goal_complete': False, 'case_pairs': cross['case_pairs'],
            'prefill': cross['prefill'], 'length_position_groups': groups,
            'within_topology_repeat_noise': noise, 'first_round_cases': cases,
            'position_statistics': positions,
            'short_cases_with_error_gt_0_1': len(early),
            'conclusion': ('Cross-topology drift already occurs below the DP8 256-token '
                           'prefill chunk size; a long-input chunk-boundary-only explanation '
                           'is insufficient. This does not identify the responsible layer '
                           'or prove TP arithmetic, attention or GDN as the sole cause.'),
            'scope': 'Read-only sealed fixed-input prefill; no serving HTTP or GPU work. '
                     'No decode comparison on differing continuation paths. '
                     'Existing numerical gates and failures remain unchanged.',
            'sources_sha256': {name: sha(root / name) for name in (
                'audit_topology_original_r99.py', 'audit_tp8_vs_attention_dp8_r139.py',
                'audit_prefill_localization_r141.py', 'tp8-vs-attention-dp8-prefill-r139.json')},
            'reference_seals': cross['reference_seals']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    assert result['short_cases_with_error_gt_0_1'] > 0
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('position_statistics', 'first_round_cases')}))
