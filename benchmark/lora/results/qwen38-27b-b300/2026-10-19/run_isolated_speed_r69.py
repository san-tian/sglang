"""Dedicated-machine final speed controls after all topology/PD GPU leases finish."""
from host_common_r50 import *
import threading
import traceback

IMAGE = 'sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7'
source = json.loads((R / 'source-host-r69.json').read_text())
for file, sha in source['source_files'].items():
    assert hashlib.sha256((R / file).read_bytes()).hexdigest() == sha, file
assert subprocess.check_output(['docker', 'image', 'inspect', 'qwen38-mol:r34', '--format', '{{.Id}}'], text=True).strip() == IMAGE
waitleases(['pd4-precision-perf-complete-r33.json', 'eight-gpu-suite-complete-r31.json',
            'multigpu-base-complete-r41.json', 'cache-extra-finished-r39.json',
            'fa4-suite-finished-r44.json', 'cached-topologies-finished-r50.json'], list(range(8)), timeout=172800)
for file in ['cache-mixed-finished-r35.json', 'cache-extra-finished-r39.json']:
    assert json.loads((R / file).read_text())['passed'], file
stop_monitor = threading.Event()
monitor_errors = []
owned = None

def monitor():
    with (R / 'isolated-speed-gpu-observations-r69.jsonl').open('x') as f:
        while not stop_monitor.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits'], text=True)
                entries = []
                for line in raw.splitlines():
                    fields = [value.strip() for value in line.split(',')]
                    entries.append({'index': int(fields[0]), 'uuid': fields[1], 'memory_mib': int(fields[2]),
                                    'utilization_pct': fields[3], 'temperature_c': fields[4],
                                    'power_w': fields[5], 'sm_clock_mhz': fields[6]})
                assert len(entries) == 8
                unexpected = [row for row in entries if row['index'] != 2 and (row['memory_mib'] >= 2000 or int(row['utilization_pct']) != 0)]
                row = {'unix_s': time.time(), 'gpus': entries, 'unexpected_active_gpus': unexpected}
                f.write(json.dumps(row) + '\n'); f.flush()
                if unexpected:
                    monitor_errors.append(row)
            except Exception:
                monitor_errors.append({'unix_s': time.time(), 'error': traceback.format_exc()})
            stop_monitor.wait(1)

thread = threading.Thread(target=monitor)
thread.start()
configs = []
result = {'passed': False, 'scope': 'GPU2/R34/mem.52/TP1/BF16 deterministic eager/Triton, identical KV65536/SSM256. Every one-second sample must show the other seven GPUs below2GiB and at0% utilization. Uncached/cached/M/base/cached/uncached order; two complete cache-control runs in opposite order. Natural matrix two warmups, three cold/hot rounds, 636 requests/16 points/run. No profiles during timing. Cold projection fill measured separately with KV flushed every request.'}
try:
    for position, side in enumerate(['nocache', 'cached', 'm', 'base', 'cached', 'nocache']):
        assert not monitor_errors, monitor_errors[-1:]
        name = f'qwen38-isolated-{position}-{side}-r69'
        model = MERGES + 'real_r16' if side == 'm' else BASE + '/' if side == 'base' else BASE
        extra = ['--max-total-tokens', '65536']
        if side == 'cached':
            extra += ['--lora-merged-bf16-cache-size-mb', '36864']
        launch(name, [2], 1, 31200, model, extra=extra, memory=.52)
        owned = name
        try:
            health(31200, name)
            with urllib.request.urlopen('http://127.0.0.1:31200/get_server_info') as f:
                info = json.load(f)
            assert info['mem_fraction_static'] == .52
            assert info['max_total_num_tokens'] == 65536 and info['max_mamba_cache_size'] == 256, info
            assert info['lora_merged_bf16_cache_size_mb'] == (36864 if side == 'cached' else 0)
            configs.append({'position': position, 'side': side, 'config': info})
            (R / 'isolated-resolved-configs-r69.json').write_text(json.dumps(configs, indent=2))
            out = f'perf-isolated-{position}-{side}-r69'
            cmd = ['docker', 'exec', name, 'python3', W + '/perf_endpoint_r29.py', '--url', 'http://127.0.0.1:31200', '--output', W + '/' + out]
            if side in ['cached', 'nocache']:
                cmd += ['--adapter-path', ADAPTERS + 'real_r16']
            run(cmd, out + '.log')
            assert not monitor_errors, monitor_errors[-1:]
            if position < 4:
                fill = f'fill-isolated-{position}-{side}-r69'
                cmd = ['docker', 'exec', '-e', 'PYTHONPATH=' + CLIENT, name, 'python3', W + '/perf_cache_fill_r47.py', '--url', 'http://127.0.0.1:31200', '--output', W + '/' + fill]
                if side in ['cached', 'nocache']:
                    cmd += ['--adapter-path', ADAPTERS + 'real_r16']
                run(cmd, fill + '.log')
        finally:
            stop([name]); owned = None
        print('completed', position, side, flush=True)
    stop_monitor.set(); thread.join(30)
    assert not thread.is_alive() and not monitor_errors, monitor_errors[-1:]
    run(['python3', str(R / 'audit_isolated_speed_r69.py')], 'isolated-speed-audit-r69.log')
    result.update(passed=True, matrix_runs=6, matrix_requests=3816, fill_requests=40,
                  artifact='isolated-speed-comparison-r69.json')
except Exception:
    result['error'] = traceback.format_exc()
    raise
finally:
    if owned:
        stop([owned])
    stop_monitor.set(); thread.join(30)
    result['monitor_errors'] = monitor_errors
    if monitor_errors:
        result['passed'] = False
    (R / 'isolated-speed-finished-r69.json').write_text(json.dumps(result, indent=2))
