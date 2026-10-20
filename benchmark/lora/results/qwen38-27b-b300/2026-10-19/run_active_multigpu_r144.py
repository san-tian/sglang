"""Dedicated TP4/attention-DP4 R55 active-cache validation after existing leases.

Five matched natural controls per topology: independent merge, base, no cache,
LRU and active. Full precision/mixed gates cover LRU and active separately.
"""
from host_common_r144 import *
import threading
import traceback

IMAGE = 'sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
SOURCE = R / 'source-host-r144.json'
ARM_ORDER = ('m', 'base', 'nocache', 'lru', 'active')


def flags(top, side):
    values = ['--max-total-tokens', '65536']
    if side in ('lru', 'active'):
        budget = 12288 if top == 'dpa4' else 10240
        values += ['--lora-merged-bf16-cache-size-mb', str(budget),
                   '--lora-merged-bf16-cache-policy', side]
    return values


def verify():
    source = json.loads(SOURCE.read_text())
    assert source['image'] == IMAGE
    for name, value in source['source_files'].items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == value, name
    return source


def wait_predecessors(source):
    # A physical resource lease, deliberately independent of numerical success.
    # All original tests keep their existing order and source/acceptance checks.
    deadline = time.monotonic() + 172800
    while True:
        live = []
        for item in source['predecessors']:
            proc = Path('/proc') / str(item['pid'])
            if not proc.exists():
                continue
            stat = (proc / 'stat').read_text().split()
            if stat[2] == 'Z':
                continue
            assert stat[21] == item['start_ticks'], 'PID reused; do not infer lease ownership'
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            assert cmd == item['cmdline'], 'Predecessor identity changed'
            live.append(item['pid'])
        if not live:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                            '--format=csv,noheader'], text=True)
            if not apps.strip():
                with (R / 'active-multigpu-resource-acquired-r144.json').open('x') as f:
                    json.dump({'unix_s': time.time(), 'predecessor_pids': [v['pid'] for v in source['predecessors']],
                               'observed_gpu_processes': apps, 'resource_only': True,
                               'prior_tests_accuracy_accepted': False, 'full_goal_complete': False}, f, indent=2)
                return
        if time.monotonic() > deadline:
            raise TimeoutError('Resource observation expired; no predecessor is stopped or restarted')
        time.sleep(5)


def observe_placement(name):
    import watch_actual_placement_r92 as observer
    observer.IMAGE = IMAGE
    observer.expected = lambda _: set(range(4))
    mapping = {v[1].strip(): int(v[0]) for v in [s.split(',') for s in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines()]}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    apps = [{'pid': int(v[0]), 'gpu_uuid': v[1].strip(), 'memory_mib': v[2].strip()}
            for v in [s.split(',') for s in raw.splitlines()] if len(v) == 3]
    info = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
    top = subprocess.check_output(['docker', 'top', name, '-eo', 'pid,ppid,args'], text=True)
    row = observer.inspect_placement(name, mapping, apps, info, top)
    assert row['placement_matched'] and len({v['pid'] for v in row['scheduler_gpu_processes']}) == 4
    with (R / 'actual-placement-observations-r144.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    return row


def frozen(name, port, top, stage, label, policy=None):
    adapter = 'real_r16' if label == 'full' else 'gdn_fused'
    directory = f'active-{top}-{label}-reference-r144' if stage == 'reference' else f'active-{top}-{policy}-{label}-dynamic-r144'
    command = ['docker', 'exec', '-e', 'PYTHONPATH=' + CLIENT, name, 'python3',
               CLIENT + '/compare_frozen.py', stage, '--url', f'http://127.0.0.1:{port}',
               '--topology', top + '-r30', '--provenance', W + '/source-host-r144.json',
               '--output', W + '/' + directory]
    if stage == 'reference':
        command += ['--corpus', '/work/qwen38-mol-experiment/phase1/corpus.json' if label == 'full' else
                    '/work/qwen38-mol-speed-20261019/corpus-mixed.json',
                    '--limit', '30' if label == 'full' else '4', '--decode-tokens', '256' if label == 'full' else '16',
                    '--rounds', '4' if label == 'full' else '1', '--batch-size', '30' if label == 'full' else '4']
    else:
        command += ['--reference', W + f'/active-{top}-{label}-reference-r144',
                    '--adapter-path', ADAPTERS + adapter]
    run(command, directory + '.log')
    result = json.loads((R / directory / 'summary.json').read_text())
    assert result['case_pairs'] == (240 if label == 'full' else 8)
    if stage == 'dynamic':
        assert all(v['passes_logprob_gate'] for v in result['scores'].values())
    return result


def natural(name, port, top, side):
    out = f'perf-active-{top}-{side}-r144'
    command = ['docker', 'exec', name, 'python3', W + '/perf_endpoint_r85.py',
               '--url', f'http://127.0.0.1:{port}', '--output', W + '/' + out, '--profile']
    if side in ('nocache', 'lru', 'active'):
        command += ['--adapter-path', ADAPTERS + 'real_r16']
    run(command, out + '.log')
    assert json.loads((R / out / 'complete.json').read_text()) == {'requests': 636, 'profile': True}
    run(['python3', str(R / 'analyze_profiles_r30.py'), str(R / out / 'profile'),
         '--output', str(R / out / 'kernel-summary.json')])


def mixed(name, port, top, policy):
    label = f'active-{top}-{policy}-mixed-r144'
    config = R / (label + '-adapters.json')
    with config.open('x') as f:
        json.dump([{'name': n, 'path': ADAPTERS + a} for n, a in
                   [('real', 'real_r16'), ('zero', 'zero'), ('gdn', 'gdn_fused')]], f)
    run(['docker', 'exec', '-e', 'PYTHONPATH=' + CLIENT, name, 'python3', CLIENT + '/compare_mixed.py',
         '--url', f'http://127.0.0.1:{port}', '--adapters', W + '/' + config.name,
         '--corpus', '/work/qwen38-mol-speed-20261019/corpus-mixed.json',
         '--capture-dir', W + f'/active-{top}-{policy}-mixed-captures-r144', '--output', W + '/' + label,
         '--cases', '4', '--decode-tokens', '16', '--rounds', '3'], label + '.log')
    summary = json.loads((R / label / 'summary.json').read_text())
    assert summary['mixed_requests'] == 48 and sum(v['count'] for v in summary['scores'].values()) == 768
    assert all(v['passes_logprob_gate'] for v in summary['scores'].values())


def main():
    source = verify()
    wait_predecessors(source)
    verify()
    end_monitor = threading.Event()
    errors = []
    observations = []
    observation_lock = threading.Lock()

    def monitor():
        with (R / 'active-multigpu-observations-r144.jsonl').open('x') as out:
            while not end_monitor.is_set():
                try:
                    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                                                   '--format=csv,noheader,nounits'], text=True)
                    rows = [[v.strip() for v in line.split(',')] for line in raw.splitlines()]
                    assert len(rows) == 8
                    unexpected = [v for v in rows if int(v[0]) >= 4 and (int(v[1]) >= 2000 or int(v[2]) != 0)]
                    row = {'unix_s': time.time(), 'rows': rows, 'unexpected': unexpected}
                    if unexpected:
                        errors.append(row)
                except BaseException:
                    row = {'unix_s': time.time(), 'error': traceback.format_exc()}
                    errors.append(row)
                out.write(json.dumps(row) + '\n')
                out.flush()
                with observation_lock:
                    observations.append(row)
                end_monitor.wait(1)

    thread = threading.Thread(target=monitor)
    thread.start()
    owned = None
    result = {'passed': False, 'full_goal_complete': False,
              'scope': source['scope'], 'runtime_image': IMAGE}
    try:
        for top, dpa, port in [('tp4', False, 31600), ('dpa4', True, 31602)]:
            for adapter, label in [('real_r16', 'full'), ('gdn_fused', 'gdn')]:
                verify()
                assert not errors and shutil.disk_usage('/').free > 4 * 1024**3
                owned = f'qwen38-active-{top}-m-{adapter}-r144'
                launch(owned, list(range(4)), 4, port, MERGES + adapter, dpa=dpa, extra=flags(top, 'm'))
                try:
                    health(port, owned)
                    observe_placement(owned)
                    frozen(owned, port, top, 'reference', label)
                    if label == 'full':
                        natural(owned, port, top, 'm')
                finally:
                    stop([owned])
                    owned = None
            for side in ARM_ORDER[1:]:
                verify()
                assert not errors and shutil.disk_usage('/').free > 4 * 1024**3
                model = BASE + '/' if side == 'base' else BASE
                owned = f'qwen38-active-{top}-{side}-r144'
                launch(owned, list(range(4)), 4, port, model, dpa=dpa, extra=flags(top, side))
                try:
                    health(port, owned)
                    observe_placement(owned)
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/get_server_info') as f:
                        info = json.load(f)
                    cfg = info.get('server_args', info)
                    assert cfg['max_total_num_tokens'] == 65536 and cfg['max_mamba_cache_size'] == 256
                    if side in ('lru', 'active'):
                        assert cfg['lora_merged_bf16_cache_policy'] == side
                        assert cfg['lora_merged_bf16_cache_size_mb'] == (12288 if dpa else 10240)
                        for label in ('full', 'gdn'):
                            frozen(owned, port, top, 'dynamic', label, side)
                        mixed(owned, port, top, side)
                    natural(owned, port, top, side)
                finally:
                    stop([owned])
                    owned = None
                print('completed', top, side, flush=True)
            assert not errors
            with observation_lock, (R / f'active-{top}-observations-r144.json').open('x') as f:
                json.dump(observations, f)
            run(['python3', str(R / 'audit_active_multigpu_r144.py'), '--topology', top],
                f'active-{top}-audit-r144.log')
        result.update(passed=True, natural_requests=6360, precision_pairs=992,
                      mixed_requests=192, rank_traces=40)
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        try:
            if owned:
                stop([owned])
        finally:
            end_monitor.set()
            thread.join(30)
            result['monitor_errors'] = errors
            if errors or thread.is_alive():
                result['passed'] = False
            with (R / 'active-multigpu-finished-r144.json').open('x') as f:
                json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
