"""Independent offline rounding, mixed-slot routing and cache reuse controls."""

import argparse
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.lora.backend.merged_fp8_backend import (
    MergedFP8LoRABackend,
    merge_fp8_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


class TestMergedFP8(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(61)

    def test_startup_guard_names_failed_requirement(self):
        args = SimpleNamespace(
            dp_size=1,
            pp_size=1,
            enable_deterministic_inference=True,
            quantization="fp8",
            speculative_algorithm=None,
            enable_lora_overlap_loading=False,
            max_lora_rank=16,
        )
        with patch(
            "sglang.srt.lora.backend.merged_fp8_backend.cuda_graph_fully_disabled",
            return_value=True,
        ):
            MergedFP8LoRABackend._validate_server_args(args)
            args.quantization = None
            with self.assertRaisesRegex(ValueError, "quantization=fp8"):
                MergedFP8LoRABackend._validate_server_args(args)
            args._resolved_overrides = [("checkpoint", {"quantization": "fp8"})]
            MergedFP8LoRABackend._validate_server_args(args)
            self.assertIsNone(args.quantization)

    def operands(self, device="cuda"):
        raw = torch.randn(256, 256) * 0.02
        scales = (raw.reshape(2, 128, 2, 128).abs().amax((1, 3)) / 448).bfloat16()
        weight = (
            raw / scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        ).to(torch.float8_e4m3fn)
        # Two packed projections, rank 2 in a rank-4 pool. Inactive A rows
        # contain a sentinel, detecting the old pool-rank/active-rank mixup.
        a = torch.randn(8, 256) * 0.01
        a[4:] = 100
        b = torch.randn(256, 4) * 0.02
        return [x.to(device) for x in (weight, scales, a, b)]

    def test_cli_registers_explicit_backend_and_reserved_cache_budget(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        cfg = parser.parse_args(
            [
                "--model-path",
                "/unused",
                "--enable-lora",
                "--lora-backend",
                "merged_fp8",
                "--lora-merged-fp8-cache-size-mb",
                "163840",
            ]
        )
        self.assertEqual(cfg.lora_backend, "merged_fp8")
        self.assertEqual(cfg.lora_merged_fp8_cache_size_mb, 163840)
        self.assertEqual(cfg.lora_merged_fp8_cache_policy, "active")

    def test_matches_independent_cpu_export_and_preserves_base(self):
        w, s, a, b = self.operands()
        before = w.view(torch.uint8).clone()
        actual = merge_fp8_weight(w, s, a, b, 2, 2.0, [0, 128, 256])
        exact = w.cpu().float() * s.cpu().float().repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        aa, bb = a.cpu(), b.cpu()
        for j in range(2):
            exact[j * 128 : (j + 1) * 128] += 2.0 * (
                bb[j * 128 : (j + 1) * 128, :2] @ aa[j * 2 : (j + 1) * 2]
            )
        qscale = (
            (exact.reshape(2, 128, 2, 128).abs().amax((1, 3)).clamp_min(1e-12) / 448)
            .bfloat16()
            .float()
        )
        expected = (
            (exact / qscale.repeat_interleave(128, 0).repeat_interleave(128, 1))
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        self.assertTrue(
            torch.equal(
                actual.weight.cpu().view(torch.uint8), expected.view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(actual.weight_scale_inv.cpu(), qscale))
        self.assertTrue(torch.equal(w.view(torch.uint8), before))

    def test_zero_scaling_is_bitwise_identity_and_shapes_are_checked(self):
        w, s, a, b = self.operands()
        zero = merge_fp8_weight(w, s, a, b, 2, 0.0, [0, 128, 256])
        self.assertIs(zero.weight, w)
        self.assertIs(zero.weight_scale_inv, s)
        with self.assertRaises(ValueError):
            merge_fp8_weight(w, s[:1], a, b, 2, 1.0, [0, 128, 256])

    def test_mixed_adapters_cache_invalidation_and_no_base_mutation(self):
        backend = object.__new__(MergedFP8LoRABackend)
        backend.device = torch.device("cuda")
        backend.max_loras_per_batch = 3
        backend._weight_cache = OrderedDict()
        backend._weight_cache_limit = 4 << 20
        backend._weight_cache_bytes = 0
        backend._cache_policy = "active"
        backend._cache_active_slots = {0, 1}
        backend._cache_transient_reserve = 0
        backend._cache_stats = dict.fromkeys(
            [
                "hits",
                "misses",
                "hit_bytes",
                "merged_bytes",
                "admissions",
                "bypasses",
                "evictions",
                "invalidations",
                "peak_retained_bytes",
            ],
            0,
        )
        backend.global_batch_info = None
        w, s, a, b = self.operands()

        # A deterministic stand-in for the base GEMM isolates routing and
        # operand override; end-to-end inference exercises the real FP8 GEMM.
        def apply(layer, x, bias):
            dequant = (
                layer.weight.float()
                * layer.weight_scale_inv.float()
                .repeat_interleave(128, 0)
                .repeat_interleave(128, 1)
            )
            return (x.float() @ dequant.t()).to(x.dtype)

        base = SimpleNamespace(
            weight=w, weight_scale_inv=s, quant_method=SimpleNamespace(apply=apply)
        )
        layer = SimpleNamespace(
            base_layer=base,
            A_buffer=torch.stack([a, a * 2, a * 0]),
            B_buffer=torch.stack([b, b, b * 0]),
            output_offset_cpu=torch.tensor([0, 128, 256]),
            merged_bf16_adapter_present=[True, True, False],
        )
        backend.merge_groups = [
            (0, 2, 2.0, torch.tensor([0, 2], device="cuda")),
            (1, 2, 1.0, torch.tensor([1], device="cuda")),
            (2, 0, 0.0, torch.tensor([3], device="cuda")),
        ]
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        before = w.view(torch.uint8).clone()
        actual = backend.run_merged_linear(layer, x, None)
        for slot, rank, scale, rows in backend.merge_groups:
            if rank:
                q = merge_fp8_weight(
                    w,
                    s,
                    layer.A_buffer[slot],
                    layer.B_buffer[slot],
                    rank,
                    scale,
                    [0, 128, 256],
                )
                expected = apply(q, x[rows], None)
            else:
                expected = apply(base, x[rows], None)
            torch.testing.assert_close(actual[rows], expected, rtol=0, atol=0)
        misses = backend._cache_stats["misses"]
        backend.run_merged_linear(layer, x, None)
        self.assertEqual(backend._cache_stats["misses"], misses)
        self.assertEqual(backend._cache_stats["hits"], 2)
        backend.invalidate_weight_cache({0})
        self.assertEqual(len(backend._weight_cache), 1)
        self.assertTrue(torch.equal(w.view(torch.uint8), before))
        self.assertLessEqual(backend._weight_cache_bytes, backend._weight_cache_limit)


if __name__ == "__main__":
    unittest.main()
