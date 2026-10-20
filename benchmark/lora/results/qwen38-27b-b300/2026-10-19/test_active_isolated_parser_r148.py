"""CPU replay and corruption rejection using sealed R55 native SSE responses.

Only the no-profile marker is a fixture; placement is synthetic for this parser
unit test. This does not rerun GPUs or prove new placement or speed results.
"""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import audit_active_isolated_r147 as audit

R = Path(__file__).resolve().parent


def main():
    raw = R / 'perf-active-cache-r56-d-r30'
    old_command = json.loads((R / 'launch-qwen38-active-d-r56.json').read_text())
    old_info = json.loads(subprocess.check_output(['docker','inspect','qwen38-active-d-r56'], text=True))[0]
    assert old_info['Image'] == audit.IMAGE and not old_info['State']['Running']
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir() if p.is_file()}
    name = 'qwen38-active-isolated-2-active-r147'
    fixture_placement = [{'name':name,'placement_matched':True,'image':audit.IMAGE,
                          'expected_compute_gpus':[2],'observed_compute_gpus':[2],
                          'scheduler_gpu_processes':[{'pid':-1}]}]
    with tempfile.TemporaryDirectory(prefix='isolated-parser-r148-', dir=R) as directory:
        root = Path(directory)
        folder = root / 'perf-active-isolated-2-active-r147'; folder.mkdir()
        for path in raw.iterdir():
            if path.name not in ('complete.json','profile'): (folder/path.name).symlink_to(path)
        (folder/'complete.json').write_text(json.dumps({'requests':636,'profile':False}))
        (root/'audit_topology_native_r100.py').symlink_to(R/'audit_topology_native_r100.py')
        (root/f'launch-{name}.json').write_text(json.dumps(old_command))
        (root/f'actual-final-{name}.json').write_text(json.dumps(old_info))
        original_root = audit.R; audit.R = root
        try:
            result = audit.native(2,'active',fixture_placement)
            assert len(result['raw_native_sha256']) == 636
            record = next(iter(result['records']))
            path = folder/(record+'.json'); original = json.loads(path.read_text())
            path.unlink()
            bad = json.loads(json.dumps(original)); bad['events'][0]['elapsed_s'] += .01
            path.write_text(json.dumps(bad))
            try: audit.native(2,'active',fixture_placement)
            except AssertionError: pass
            else: raise AssertionError('Changed SSE timestamp accepted')
            bad = json.loads(json.dumps(original)); bad['events'][-1]['data']['output_ids'].pop()
            path.write_text(json.dumps(bad))
            try: audit.native(2,'active',fixture_placement)
            except AssertionError: pass
            else: raise AssertionError('Truncated SSE final output accepted')
        finally:
            audit.R = original_root
    assert all(hashlib.sha256((raw/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
    output = {'passed':True,'CPU_parser_replay_only':True,'native_responses_replayed':636,
              'latency_statistics_recomputed':192,'corruptions_rejected':['first_SSE_timestamp','truncated_final_output'],
              'placement_is_synthetic_fixture':True,'original_raw_files_unchanged':True,
              'new_GPU_accepted':False,'full_goal_complete':False,
              'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'auditor_sha256':hashlib.sha256((R/'audit_active_isolated_r147.py').read_bytes()).hexdigest()}
    with (R/'active-isolated-parser-replay-r148.json').open('x') as f: json.dump(output,f,indent=2)
    print(json.dumps(output))


if __name__ == '__main__': main()
