"""CPU checks of diagnostic placement and cleanup after partial pair startup."""
import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parent
helper = ast.parse((root / 'host_common_r80.py').read_text())
launch = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'launch')
driver = ast.parse((root / 'run_pd2_diagnostic_r80.py').read_text())
pair = next(n for n in driver.body if isinstance(n, ast.FunctionDef) and n.name == 'pair')
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    (folder / 'launch-qwen38-tp2-d-r28.json').write_bytes(
        (root / 'launch-qwen38-tp2-d-r28.json').read_bytes())
    ns = {'R': folder, 'json': json, 'BASE': '/work/models/Qwen3.8-27B',
          'shutil': SimpleNamespace(copytree=lambda *a: None, copyfile=lambda *a: None),
          'CLIENT': '/work/qwen38-mol-multigpu-20261019/client-r33', 'run': lambda *a: None}
    exec(compile(ast.Module(body=[launch], type_ignores=[]), '<launch>', 'exec'), ns)
    count = 0
    for label, model in [('d', ns['BASE']), ('m-real', '/work/merged/real'), ('m-gdn', '/work/merged/gdn')]:
        for index, gpus in enumerate(([0, 1], [2, 3])):
            name = f'check-{label}-{index}'
            ns['launch'](name, gpus, 2, 31500 + index, model)
            cmd = json.loads((folder / f'launch-{name}.json').read_text())
            assert cmd[cmd.index('--gpus') + 1] == '"device=0,3,4,5"'
            assert cmd[cmd.index('--pid') + 1] == 'host'
            assert cmd[cmd.index('--base-gpu-id') + 1] == str(index * 2)
            assert cmd[cmd.index('--tp-size') + 1] == '2'
            assert ('--enable-lora' in cmd) == (label == 'd')
            count += 1
    calls, stopped = [], []
    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise RuntimeError('injected second-role startup failure')
    ns.update(guard=lambda: None, launch=fail_second,
              cleanup=lambda names, label: stopped.append(names),
              portp=31500, portd=31501, bootstrap=31510, env={})
    exec(compile(ast.Module(body=[pair], type_ignores=[]), '<pair>', 'exec'), ns)
    try:
        ns['pair']('d', ns['BASE'], lambda names: None)
    except RuntimeError as error:
        assert 'injected' in str(error)
    else:
        raise AssertionError('failure swallowed')
    assert stopped == [calls] and len(calls) == 2
    print(json.dumps({'launch_cases': count, 'partial_startup_cleanup': 'passed',
                      'docker_invocations': 0, 'gpu_acceptance': False}))
