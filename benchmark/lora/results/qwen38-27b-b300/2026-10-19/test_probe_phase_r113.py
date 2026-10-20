import tempfile
import unittest
from pathlib import Path

from watch_dcp_kernel_probe_r113 import phase_ready, precision_active


class ProbeWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.phase = self.root/'dcp8x2-full-dynamic-r30'
        self.phase.mkdir()

    def test_reference_or_initialization_is_not_a_probe_window(self):
        self.assertFalse(phase_ready(self.root))

    def test_early_precision_allows_probe(self):
        (self.phase/'contract.json').write_text('{}')
        self.assertTrue(phase_ready(self.root))
        (self.phase/'cases.jsonl').write_text('{}\n'*30)
        self.assertTrue(phase_ready(self.root))

    def test_later_precision_does_not_allow_a_new_probe(self):
        (self.phase/'contract.json').write_text('{}')
        (self.phase/'cases.jsonl').write_text('{}\n'*60)
        self.assertFalse(phase_ready(self.root))
        self.assertTrue(precision_active(self.root))

    def test_completed_precision_is_not_a_probe_window(self):
        (self.phase/'contract.json').write_text('{}')
        (self.phase/'summary.json').write_text('{}')
        self.assertFalse(phase_ready(self.root))

    def test_native_timing_never_allows_probe(self):
        (self.phase/'contract.json').write_text('{}')
        native = self.root/'perf-dcp8x2-d-r30'
        native.mkdir()
        (native/'requests.jsonl').touch()
        self.assertFalse(phase_ready(self.root))
        self.assertFalse(precision_active(self.root))


if __name__ == '__main__':
    unittest.main()
