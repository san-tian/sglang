"""Matched R55 single-GPU cache controls after all original and R144 GPU owners."""
from host_common_r144 import *
import threading
import traceback

IMAGE = 'sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
ORDER = ('nocache', 'lru', 'active', 'm', 'base', 'active', 'lru', 'nocache')


def flags(side):
    values = ['--max-total-tokens', '65536']
    if side in ('lru', 'active'):
        values += ['--lora-merged-bf16-cache-size-mb', '36864',
                   '--lora-merged-bf16-cache-policy', side]
    return values


def verify():
    source = json.loads((R / 'source-host-r147.json').read_text())
    assert source['runtime_image'] == IMAGE
    for name, value in source['source_files'].items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == value, name
    return source


def wait_predecessors(source):
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
                with (R / 'active-isolated-resource-acquired-r147.json').open('x') as f:
                    json.dump({'unix_s': time.time(), 'resource_only': True,
                               'prior_tests_accuracy_accepted': False,
                               'full_goal_complete': False, 'observed_gpu_processes': apps}, f)
                return
        if time.monotonic() > deadline:
            raise TimeoutError('Resource observation expired; no predecessor is stopped or restarted')
        time.sleep(5)


def placement(name):
    import watch_actual_placement_r92 as observer
    observer.IMAGE = IMAGE
    observer.expected = lambda _: {2}
    gpus = {v[1].strip(): int(v[0]) for v in [s.split(',') for s in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines()]}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    apps = [{'pid': int(v[0]), 'gpu_uuid': v[1].strip(), 'memory_mib': v[2].strip()}
            for v in [s.split(',') for s in raw.splitlines()] if len(v) == 3]
    info = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
    top = subprocess.check_output(['docker', 'top', name, '-eo', 'pid,ppid,args'], text=True)
    row = observer.inspect_placement(name, gpus, apps, info, top)
    assert row['placement_matched'] and len({v['pid'] for v in row['scheduler_gpu_processes']}) == 1
    with (R / 'actual-placement-observations-r147.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def main():
    source = verify()
    wait_predecessors(source)
    verify()
    stopping = threading.Event()
    errors = []
    owned = None
    result = {'passed': False, 'full_goal_complete': False, 'scope': source['scope']}

    def monitor():
        with (R / 'active-isolated-observations-r147.jsonl').open('x') as f:
            while not stopping.is_set():
                try:
                    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits'], text=True)
                    rows = [[v.strip() for v in s.split(',')] for s in raw.splitlines()]
                    assert len(rows) == 8
                    unexpected = [v for v in rows if int(v[0]) != 2 and (int(v[1]) >= 2000 or int(v[2]) != 0)]
                    row = {'unix_s': time.time(), 'rows': rows, 'unexpected': unexpected}
                    if unexpected:
                        errors.append(row)
                except BaseException:
                    row = {'unix_s': time.time(), 'error': traceback.format_exc()}
                    errors.append(row)
                f.write(json.dumps(row) + '\n'); f.flush()
                stopping.wait(1)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        for position, side in enumerate(ORDER):
            verify()
            assert not errors and shutil.disk_usage('/').free > 4 * 1024**3
            name = f'qwen38-active-isolated-{position}-{side}-r147'
            model = MERGES + 'real_r16' if side == 'm' else BASE + '/' if side == 'base' else BASE
            owned = name
            launch(name, [2], 1, 31700, model, memory=.52, extra=flags(side))
            try:
                health(31700, name)
                placement(name)
                cmd = ['docker', 'exec', name, 'python3', W + '/perf_endpoint_r85.py',
                       '--url', 'http://127.0.0.1:31700', '--output', W + f'/perf-active-isolated-{position}-{side}-r147']
                if side in ('nocache', 'lru', 'active'):
                    cmd += ['--adapter-path', ADAPTERS + 'real_r16']
                run(cmd, f'perf-active-isolated-{position}-{side}-r147.log')
                if position < 5:
                    cmd = ['docker', 'exec', '-e', 'PYTHONPATH=' + CLIENT, name, 'python3', W + '/perf_cache_fill_r47.py',
                           '--url', 'http://127.0.0.1:31700', '--output', W + f'/' + f'fill-active-isolated-{position}-{side}-r147']
                    if side in ('nocache', 'lru', 'active'):
                        cmd += ['--adapter-path', ADAPTERS + 'real_r16']
                    run(cmd, f'fill-active-isolated-{position}-{side}-r147.log')
            finally:
                stop([name]); owned = None
                actual = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
                with (R / f'actual-final-{name}.json').open('x') as f:
                    json.dump(actual, f)
            assert not errors
            print('completed', position, side, flush=True)
        stopping.set(); thread.join(30)
        assert not errors and not thread.is_alive()
        run(['python3', str(R / 'audit_active_isolated_r147.py')], 'active-isolated-audit-r147.log')
        result.update(passed=True, natural_requests=5088, fill_requests=50,
                      report='active-isolated-audit-r147.json')
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        try:
            if owned:
                stop([owned])
        finally:
            stopping.set(); thread.join(30)
        result['monitor_errors'] = errors
        if errors or thread.is_alive():
            result['passed'] = False
        with (R / 'active-isolated-finished-r147.json').open('x') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
