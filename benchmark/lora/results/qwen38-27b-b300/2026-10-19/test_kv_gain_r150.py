"""CPU replay of real TP8 KV baseline plus explicit gain-gate fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import audit_kv_gain_r150 as audit

R = Path(__file__).resolve().parent


def main():
    baseline = audit.reconstruct(R,'tp8')
    c8 = next(c for c in baseline['cases'] if c['http_concurrency']==8)
    assert c8['peak']['normalized']['totals']['kv_used_tokens'] == 32545
    assert c8['peak']['normalized']['totals']['num_running_reqs'] == 5
    assert not c8['sustained_full_batch']
    larger = copy.deepcopy(baseline)
    larger['logical_configured_rows'] *= 2
    assert audit.gain(baseline,larger)['passed'] is False
    fixture = next(c for c in larger['cases'] if c['http_concurrency']==8)
    fixture['peak']['normalized']['totals']['kv_used_tokens'] = 52000
    assert audit.gain(baseline,larger)['passed'] is False
    fixture['sustained_full_batch'] = True
    assert audit.gain(baseline,larger)['passed'] is True
    report = {'passed':True,'CPU_replay_only':True,'real_TP8_native_requests_reparsed':13,
              'real_TP8_metrics_snapshots_reparsed':baseline['raw_snapshot_count'],
              'real_TP8_c8_observed_KV_tokens':32545,'real_TP8_c8_observed_running_requests':5,
              'negative_fixtures_rejected':['larger_configured_pool_without_observed_gain','larger_peak_without_sustained_full_batch'],
              'synthetic_positive_fixture_passed':True,'fixtures_are_not_GPU_evidence':True,
              'new_DP8_capacity_accepted':False,'full_goal_complete':False,
              'original_TP8_report_sha256':baseline['original_report_sha256'],
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'auditor_sha256':hashlib.sha256((R/'audit_kv_gain_r150.py').read_bytes()).hexdigest()}
    with (R/'kv-gain-cpu-replay-r150.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__': main()
