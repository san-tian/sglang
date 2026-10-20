"""Read-only wait for the actual R135 merged native producer, then audit it."""
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
producer = json.loads((R / 'attention-dp8-process-r135.json').read_text())
assert producer['source_sha256'] == hashlib.sha256((R / 'run_attention_dp8_r135.py').read_bytes()).hexdigest()
result = {'passed': False, 'full_goal_complete': False, 'producer_pid': producer['pid'],
          'scope': 'Merged native controls only; numerical parity, MoL, capacity and full goal remain separate.'}
try:
    complete = R / 'perf-dpa8-m-r135/complete.json'
    deadline = time.monotonic() + 172800
    while not complete.exists():
        proc = Path('/proc') / str(producer['pid'])
        assert proc.exists() and (proc / 'stat').read_text().split()[2] != 'Z', 'Producer terminated before native completion'
        assert b'run_attention_dp8_r135.py' in (proc / 'cmdline').read_bytes()
        if time.monotonic() > deadline:
            raise TimeoutError('Observation expired; never restart producer solely on this timeout')
        time.sleep(5)
    with (R / 'merge-baselines-audit-host-r142.log').open('x') as stream:
        subprocess.run(['python3', str(R / 'audit_merge_baselines_r142.py'), '--output',
                        str(R / 'merge-baselines-audit-r142.json')], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    report = R / 'merge-baselines-audit-r142.json'
    assert json.loads(report.read_text())['passed']
    result.update(passed=True, report=report.name,
                  report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
except BaseException:
    result['error'] = traceback.format_exc()
    raise
finally:
    result['unix_s'] = time.time()
    with (R / 'merge-baselines-observer-finished-r142.json').open('x') as stream:
        json.dump(result, stream, indent=2)
