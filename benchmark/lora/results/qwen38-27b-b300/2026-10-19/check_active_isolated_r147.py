"""CPU checks of real launch construction and resource-lease rejection paths."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
driver = importlib.import_module('run_active_isolated_r147')
helper = importlib.import_module('host_common_r144')


class Controls(unittest.TestCase):
    def test_eight_actual_launches(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'launch-qwen38-tp2-d-r28.json').write_bytes((ROOT / 'launch-qwen38-tp2-d-r28.json').read_bytes())
            with patch.object(helper, 'R', root), patch.object(helper, 'run', lambda command, *args: calls.append(command)), \
                 patch('shutil.copytree'), patch('shutil.copyfile'):
                for position, side in enumerate(driver.ORDER):
                    name = f'check-{position}-{side}'
                    model = helper.MERGES + 'real_r16' if side == 'm' else helper.BASE + '/' if side == 'base' else helper.BASE
                    helper.launch(name, [2], 1, 31700, model, memory=.52, extra=driver.flags(side))
                    command = calls[-2]
                    self.assertIn(driver.IMAGE, command)
                    for flag, value in [('--gpus', '\"device=2\"'), ('--tp-size', '1'), ('--model-path', model),
                                        ('--max-total-tokens', '65536'), ('--max-mamba-cache-size', '256'),
                                        ('--mem-fraction-static', '0.52')]:
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertNotIn('--attn-dp-size', command)
                    self.assertEqual('--enable-lora' in command, side in ('nocache', 'lru', 'active'))
                    self.assertEqual('--lora-merged-bf16-cache-policy' in command, side in ('lru', 'active'))
                    if side in ('lru', 'active'):
                        self.assertEqual(command[command.index('--lora-merged-bf16-cache-policy') + 1], side)
                        self.assertEqual(command[command.index('--lora-merged-bf16-cache-size-mb') + 1], '36864')
                    self.assertIn('TMPDIR=/root/.cache/tmp', command)
                    self.assertEqual(calls[-1], ['docker', 'start', name])
                self.assertEqual(len(calls), 16)

    def test_terminal_predecessor_is_resource_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(driver.subprocess, 'check_output', return_value=''):
            root = Path(directory)
            with patch.object(driver, 'R', root):
                driver.wait_predecessors({'predecessors': []})
            report = json.loads((root / 'active-isolated-resource-acquired-r147.json').read_text())
            self.assertTrue(report['resource_only'])
            self.assertFalse(report['prior_tests_accuracy_accepted'])
            self.assertFalse(report['full_goal_complete'])

    def test_busy_gpu_never_acquires(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(driver.subprocess, 'check_output', return_value='999, GPU-busy\n'), \
             patch.object(driver.time, 'monotonic', side_effect=[0, 172801]), patch.object(driver.time, 'sleep'):
            root = Path(directory)
            with patch.object(driver, 'R', root), self.assertRaises(TimeoutError):
                driver.wait_predecessors({'predecessors': []})
            self.assertFalse((root / 'active-isolated-resource-acquired-r147.json').exists())

    def test_reused_live_pid_rejected(self):
        import os
        proc = Path('/proc') / str(os.getpid())
        row = {'pid': os.getpid(), 'start_ticks': '-1', 'cmdline': 'incorrect'}
        self.assertTrue(proc.exists())
        with self.assertRaisesRegex(AssertionError, 'PID reused'):
            driver.wait_predecessors({'predecessors': [row]})


if __name__ == '__main__':
    unittest.main()
