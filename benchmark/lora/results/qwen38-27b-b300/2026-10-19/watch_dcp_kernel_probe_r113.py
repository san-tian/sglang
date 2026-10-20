"""Run a small additive GPU diagnostic only during early DCP precision, never timing."""
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
W = '/work/qwen38-mol-multigpu-20261019'


def precision_active(root):
    directory = root/'dcp8x2-full-dynamic-r30'
    if not (directory/'contract.json').exists() or (directory/'summary.json').exists():
        return False
    if (root/'perf-dcp8x2-d-r30/requests.jsonl').exists():
        return False
    return True


def phase_ready(root):
    if not precision_active(root):
        return False
    rows = root/'dcp8x2-full-dynamic-r30/cases.jsonl'
    return not rows.exists() or len(rows.read_text().splitlines()) <= 30


def main():
    result = {'passed': False, 'full_goal_complete': False}
    try:
        deadline = time.monotonic()+172800
        name = 'qwen38-dcp8x2-d-r31'
        while not phase_ready(R):
            p = Path('/proc/954883')
            assert p.exists() and (p/'stat').read_text().split()[2] != 'Z'
            assert b'run_eight_gpu_suite_r85.py' in (p/'cmdline').read_bytes()
            if (R/'dcp8x2-full-dynamic-r30/summary.json').exists():
                raise RuntimeError('Early precision window missed; no GPU diagnostic started')
            assert time.monotonic() < deadline, 'Observation expired; do not restart GPU workers'
            time.sleep(5)
        assert phase_ready(R)
        state = json.loads(subprocess.check_output(['docker','inspect',name], text=True))[0]
        assert state['State']['Running'] and state['Image'] == 'sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
        placement = [json.loads(line) for line in (R/'actual-placement-observations-r92.jsonl').read_text().splitlines()]
        assert any(p['name'] == name and p['placement_matched'] for p in placement)
        free = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True)
        assert int(free.strip()) > 4096
        result.update(container_id=state['Id'], image=state['Image'], started_unix_s=time.time(), phase_before='early DCP full dynamic precision', gpu=0)
        cmd = ['docker','exec','-e','CUDA_VISIBLE_DEVICES=0','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
               name,'timeout','--kill-after=5','75','python3',W+'/probe_dcp_extend_kernel_r113.py','--output',W+'/dcp-extend-kernel-probe-r113.json']
        result['command'] = cmd
        with (R/'dcp-extend-kernel-probe-r113.log').open('x') as out:
            completed = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=90)
        result['returncode'] = completed.returncode
        assert precision_active(R), 'Diagnostic crossed into another phase; preserve this failure'
        data = json.loads((R/'dcp-extend-kernel-probe-r113.json').read_text())
        result.update(passed=data['passed'], operator_result_sha256=hashlib.sha256((R/'dcp-extend-kernel-probe-r113.json').read_bytes()).hexdigest())
        assert completed.returncode == 0 and data['passed'], data
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['finished_unix_s'] = time.time()
        with (R/'dcp-kernel-probe-finished-r113.json').open('x') as out:
            json.dump(result, out, indent=2)


if __name__ == '__main__':
    main()
