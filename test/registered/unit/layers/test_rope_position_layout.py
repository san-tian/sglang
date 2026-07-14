import unittest

import torch

from sglang.srt.layers.rotary_embedding.utils import canonicalize_rope_positions
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRopePositionLayout(unittest.TestCase):
    def test_strided_positions_become_contiguous(self):
        positions = torch.arange(32).view(4, 8)[:, ::2]
        self.assertFalse(positions.is_contiguous())
        self.assertNotEqual(positions.stride(-1), 1)

        canonical = canonicalize_rope_positions(positions)

        self.assertTrue(canonical.is_contiguous())
        self.assertEqual(canonical.stride(-1), 1)
        torch.testing.assert_close(canonical, positions)

    def test_contiguous_positions_keep_storage(self):
        positions = torch.arange(32).view(4, 8)

        canonical = canonicalize_rope_positions(positions)

        self.assertIs(canonical, positions)


if __name__ == "__main__":
    unittest.main()
