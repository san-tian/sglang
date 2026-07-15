import ast
import types
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


SOURCE = Path(__file__).parents[4] / "python/sglang/srt/models/deepseek_v2.py"


def load_rebuild_method():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2AttentionMLA":
            method = next(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and item.name == "rebuild_cp_kv_cache"
            )
            break
    if method is None:
        raise AssertionError("rebuild_cp_kv_cache was not found")

    namespace = {
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(current_stream=lambda: None)
        ),
        "cp_all_gather_rerange_output": lambda cache, *_: cache,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["rebuild_cp_kv_cache"]


class TestDeepseekCpKvCacheAlias(unittest.TestCase):
    def test_overlapping_k_pe_view_is_copied_before_repack(self):
        rebuild = load_rebuild_method()
        self_obj = types.SimpleNamespace(kv_lora_rank=4, cp_size=1)

        storage = torch.arange(10, dtype=torch.float32)
        latent_cache = storage[:8].view(1, 8)
        k_nope = torch.full((1, 1, 4), -1.0)
        # The shifted view shares storage with the PE destination without
        # crossing into the k_nope region.  The unpatched assignment raises
        # the same overlap RuntimeError observed by the Prefill scheduler.
        k_pe = storage[5:9].view(1, 1, 4)
        expected_pe = k_pe.squeeze(1).clone()

        result_nope, result_pe = rebuild(self_obj, latent_cache, None, k_nope, k_pe)

        torch.testing.assert_close(latent_cache[..., :4], k_nope.squeeze(1))
        torch.testing.assert_close(latent_cache[..., 4:], expected_pe)
        torch.testing.assert_close(result_nope.squeeze(1), latent_cache[..., :4])
        torch.testing.assert_close(result_pe.squeeze(1), latent_cache[..., 4:])


if __name__ == "__main__":
    unittest.main()
