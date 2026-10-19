"""Additional real 2P2D diagnostic; never substitutes for the queued 4P4D matrix."""
from host_common_r80 import *
import traceback

IMAGE = 'sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
GPUS = [0, 3, 4, 5]
portp, portd, bootstrap = 31500, 31501, 31510
env = {'SGLANG_DISAGGREGATION_NIXL_BACKEND': 'UCX',
       'UCX_TLS': 'cuda_copy,cuda_ipc,posix,sysv,tcp', 'UCX_LOG_LEVEL': 'info'}
manifest = json.loads((R / 'source-host-r80.json').read_text())
for filename, sha in manifest['source_files'].items():
    assert hashlib.sha256((R / filename).read_bytes()).hexdigest() == sha, filename


def write(filename, value):
    with (R / filename).open('x') as f:
        json.dump(value, f, indent=2)


def guard():
    assert not (R / 'eight-gpu-suite-started-r31.json').exists()
    assert not (R / 'active-cache-candidate-finished-r56.json').exists()
    # This bounded diagnostic runs only while the existing live R56 lease
    # prevents the all-eight-GPU coordinator from acquiring these GPUs.
    assert b'run_active_cache_suite_r56.py' in Path('/proc/848016/cmdline').read_bytes()
    usage = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = {int(a): [int(b), int(c)] for a, b, c in
            (line.split(',') for line in usage.splitlines())}
    assert all(rows[g][0] < 2000 and rows[g][1] == 0 for g in GPUS), rows
    return rows


def inspect_pair(names, label):
    inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    evidence = {'gpu_inventory': inventory, 'gpu_processes': processes,
                'containers': {}, 'utc': time.time()}
    for name in names:
        item = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        assert item['Image'] == IMAGE
        assert item['HostConfig']['PidMode'] == 'host'
        assert item['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == ['0', '3', '4', '5']
        evidence['containers'][name] = {
            'image': item['Image'], 'devices': item['HostConfig']['DeviceRequests'],
            'pid_mode': item['HostConfig']['PidMode'], 'state': item['State'],
            'command': item['Config']['Cmd'],
            'host_processes': subprocess.check_output(
                ['docker', 'top', name, '-eo', 'pid,ppid,args'], text=True)}
    write(f'pd2-placement-{label}-r80.json', evidence)


def cleanup(names, label):
    # Own both role names before creating either one, so partial startup is
    # cleaned too. Only these exact diagnostic containers may be stopped.
    present = []
    for name in names:
        assert name.startswith('qwen38-pd2-') and name.endswith('-r80')
        result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
        if result.returncode == 0:
            present.append(name)
    if present:
        try:
            for name in present:
                with (R / (name + '-final.log')).open('x') as f:
                    subprocess.run(['docker', 'logs', name], stdout=f,
                                   stderr=subprocess.STDOUT, check=False)
        finally:
            # P and D always stop in the same docker invocation.
            run(['docker', 'stop', '--time', '30', *present])
        write(f'pd2-stopped-{label}-r80.json',
              json.loads(subprocess.check_output(['docker', 'inspect', *present])))


def pair(label, model, operation):
    guard()
    names = [f'qwen38-pd2-{label}-{role}-r80' for role in ('prefill', 'decode')]
    try:
        for name, mode, port, gpus, nccl in [
            (names[0], 'prefill', portp, [0, 1], 31520),
            (names[1], 'decode', portd, [2, 3], 31521)]:
            launch(name, gpus, 2, port, model, env=env, extra=[
                '--disaggregation-mode', mode,
                '--disaggregation-bootstrap-port', str(bootstrap),
                '--nccl-port', str(nccl), '--disaggregation-transfer-backend', 'nixl',
                '--mamba-radix-cache-strategy', 'extra_buffer', '--enable-metrics',
                '--disaggregation-enable-kv-checksum', '--watchdog-timeout', '900'])
        for name, port in zip(names, [portp, portd]):
            health(port, name)
        inspect_pair(names, label)
        operation(names)
    finally:
        cleanup(names, label)


def precision(names, stage, label, adapter):
    result = compare(names[1], stage, label, portd, 'pd2diag', adapter,
                     limit=4, nt=16, rounds=1, batch=4,
                     prefill=portp, bootstrap=bootstrap)
    if stage == 'dynamic':
        assert result['case_pairs'] == 8
        assert all(v['passes_logprob_gate'] for v in result['scores'].values())
    return result


def performance(names, side):
    out = f'perf-pd2diag-{side}-r80'
    cmd = ['docker', 'exec', '-e', 'PYTHONPATH=' + CLIENT, names[1],
           'python3', W + '/perf_pd_r31.py', '--prefill-url', f'http://127.0.0.1:{portp}',
           '--url', f'http://127.0.0.1:{portd}', '--bootstrap-port', str(bootstrap),
           '--output', W + '/' + out, '--smoke', '--profile']
    if side == 'd':
        cmd += ['--adapter-path', ADAPTERS + 'real_r16']
    run(cmd, out + '.log')
    run(['python3', str(R / 'analyze_profiles_r30.py'), str(R / out / 'profile'),
         '--output', str(R / out / 'kernel-summary.json')])


def main():
    write('pd2-diagnostic-started-r80.json', {'before': guard(), 'utc': time.time(),
          'scope': 'Additional 2P2D diagnostic only; full 4P4D remains mandatory.'})
    for adapter, label in [('real_r16', 'full'), ('gdn_fused', 'gdn')]:
        def reference(names, adapter=adapter, label=label):
            precision(names, 'reference', label, adapter)
            if label == 'full':
                performance(names, 'm')
        pair('m-' + adapter, MERGES + adapter, reference)

    def dynamic(names):
        for adapter, label in [('real_r16', 'full'), ('gdn_fused', 'gdn')]:
            precision(names, 'dynamic', label, adapter)
        performance(names, 'd')
    pair('d', BASE, dynamic)
    left = [json.loads(x) for x in
            (R / 'perf-pd2diag-d-r80/requests.jsonl').read_text().splitlines()]
    right = {r['label']: r for r in [json.loads(x) for x in
             (R / 'perf-pd2diag-m-r80/requests.jsonl').read_text().splitlines()]}
    assert len(left) == len(right) == 2
    assert all(r['output_ids'] == right[r['label']]['output_ids'] for r in left)
    write('pd2-diagnostic-complete-r80.json', {
        'scope': '2P2D diagnostic, not full performance or 4P4D acceptance',
        'real_case_pairs': 8, 'gdn_case_pairs': 8,
        'natural_requests_per_side': 2, 'natural_output_ids_equal': True,
        'actual_transport_and_checksums': 'Inspect original role logs and profiles; flags alone are insufficient.'})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write('pd2-diagnostic-failed-r80.json', {'utc': time.time(), 'traceback': traceback.format_exc()})
        raise
