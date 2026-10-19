"""Fresh six-policy experiment after terminal R56 ENOSPC; preserve original failures."""
from host_common_r84 import *
import traceback

manifest = json.loads((R / 'source-host-r84.json').read_text())
for filename, sha in manifest['source_files'].items():
    assert hashlib.sha256((R / filename).read_bytes()).hexdigest() == sha, filename
assert not json.loads((R / 'active-cache-candidate-finished-r56.json').read_text())['passed']
assert shutil.disk_usage('/').free > 4 * 1024**3
assert not (R / 'active-cache-recovery-finished-r84.json').exists()
waitleases([], list(range(8)), timeout=1800)
owned = None
result = {'passed': False, 'scope': 'Fresh R55 six-policy mixed matrices, 1152 natural requests; '
          'R56 main/GDN/prefix/lifecycle/native evidence unchanged. Completion alone is not speed benefit.',
          'root_safe_tmpdir': '/root/.cache/tmp', 'root_safe_pycache': '/root/.cache/pycache'}


def counters():
    raw = subprocess.check_output(['docker', 'logs', owned], stderr=subprocess.STDOUT, text=True)
    rows = [json.loads(line.split('projection_cache_stats=', 1)[1])
            for line in raw.splitlines() if 'projection_cache_stats=' in line]
    return rows[-1] if rows else None


def stage(label, cmd):
    before = counters()
    run(cmd, label + '.log')
    (R / (label + '-cache-counters.json')).write_text(
        json.dumps({'before': before, 'after': counters()}, indent=2))


def finish():
    global owned
    if owned is not None:
        name, owned = owned, None
        found = subprocess.run(['docker', 'inspect', name], capture_output=True)
        if found.returncode == 0:
            stop([name])


try:
    for index, policy in enumerate(['nocache', 'lru', 'active', 'active', 'lru', 'nocache']):
        assert shutil.disk_usage('/').free > 4 * 1024**3
        owned = f'qwen38-active-mix-{index}-{policy}-r84'
        flags = ['--max-total-tokens', '65536']
        if policy != 'nocache':
            flags += ['--lora-merged-bf16-cache-size-mb', '36864',
                      '--lora-merged-bf16-cache-policy', policy]
        launch(owned, [2], 1, 31400, BASE, memory=.52, extra=flags)
        try:
            health(31400, owned)
            with urllib.request.urlopen('http://127.0.0.1:31400/get_server_info') as f:
                cfg = json.load(f)
            assert cfg['max_total_num_tokens'] == 65536 and cfg['max_mamba_cache_size'] == 256
            actual = json.loads(subprocess.check_output(['docker', 'inspect', owned]))[0]
            assert actual['Image'] == manifest['image']
            assert 'TMPDIR=/root/.cache/tmp' in actual['Config']['Env']
            assert 'PYTHONPYCACHEPREFIX=/root/.cache/pycache' in actual['Config']['Env']
            (R / (owned + '-actual-config.json')).write_text(json.dumps({
                'image': actual['Image'], 'env': actual['Config']['Env'],
                'mounts': actual['Mounts'], 'server_info': cfg}, indent=2))
            out = f'perf-active-mix-{index}-{policy}-r84'
            stage(out, ['docker', 'exec', owned, 'python3', W + '/perf_cache_mixed_r35.py',
                        '--url', 'http://127.0.0.1:31400', '--output', W + '/' + out])
            if index < 3:
                out = f'profile-active-mix-{index}-{policy}-r84'
                stage(out, ['docker', 'exec', owned, 'python3', W + '/profile_mixed_cache_r84.py',
                            '--url', 'http://127.0.0.1:31400', '--output', W + '/' + out])
                run(['python3', str(R / 'analyze_profiles_r30.py'), str(R / out / 'profile'),
                     '--output', str(R / out / 'kernel-summary.json')])
        finally:
            finish()
    run(['python3', str(R / 'audit_active_cache_suite_r84.py')], 'active-cache-suite-audit-r84.log')
    result.update(passed=True, mixed_runs=6, natural_requests=1152)
except BaseException:
    result['error'] = traceback.format_exc()
    raise
finally:
    try:
        finish()
    finally:
        with (R / 'active-cache-recovery-finished-r84.json').open('x') as f:
            json.dump(result, f, indent=2)
