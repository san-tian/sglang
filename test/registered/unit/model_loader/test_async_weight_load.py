import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.utils import should_async_load
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestShouldAsyncLoad(unittest.TestCase):
    def test_cpu_weight_uses_async_loading_by_default(self):
        weight = SimpleNamespace(device=torch.device("cpu"))

        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(should_async_load(weight))

    def test_env_guard_disables_async_loading(self):
        weight = SimpleNamespace(device=torch.device("cpu"))

        for value in ("1", "true", "TRUE", "yes", "on"):
            with (
                self.subTest(value=value),
                patch.dict(
                    "os.environ",
                    {"SGLANG_DISABLE_ASYNC_WEIGHT_LOAD": value},
                    clear=True,
                ),
            ):
                self.assertFalse(should_async_load(weight))

    def test_false_env_value_preserves_async_loading(self):
        weight = SimpleNamespace(device=torch.device("cpu"))

        with patch.dict(
            "os.environ", {"SGLANG_DISABLE_ASYNC_WEIGHT_LOAD": "0"}, clear=True
        ):
            self.assertTrue(should_async_load(weight))

    def test_non_cpu_weight_never_uses_async_loading(self):
        weight = SimpleNamespace(device=torch.device("cuda"))

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(should_async_load(weight))


if __name__ == "__main__":
    unittest.main()
