"""Eager dense LoRA with the rounding semantics of an offline block-FP8 merge.

This is an explicit alternative to additive low-rank serving: dequantize the
immutable base, merge FP32 factors, round block scales to BF16, requantize, and
use the base FP8 GEMM. Derived weights may be cached; adapters stay low rank.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

import torch
import torch.nn.functional as F
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.lora.backend.merged_bf16_backend import (
    MergedBF16LoRABackend,
    merge_bf16_weight,
)
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled


@dataclass
class FP8Projection:
    weight: torch.Tensor
    weight_scale_inv: torch.Tensor

    def numel(self):
        # The shared projection cache accounts heterogeneous tensors in bytes.
        return sum(
            t.numel() * t.element_size() for t in (self.weight, self.weight_scale_inv)
        )

    def element_size(self):
        return 1


def merge_fp8_weight(weight, weight_scale, a, b, rank, scaling, offsets):
    """Return row-major E4M3 weights and FP32 views of BF16 block scales.

    Packed A uses the adapter's active rank, while packed B retains pool-rank
    stride. No base tensors are modified. Zero scaling preserves all bits.
    """
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("merged_fp8 requires IEEE FP32 matmul (TF32 disabled)")
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("merged_fp8 requires a matrix of E4M3 checkpoint weights")
    n, k = weight.shape
    if n % 128 or k % 128 or weight_scale.shape != (n // 128, k // 128):
        raise ValueError("merged_fp8 requires aligned 128x128 weight blocks")
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.dtype != torch.float32
        or b.dtype != torch.float32
        or not 0 < rank <= b.shape[-1] <= 64
        or a.shape != ((len(offsets) - 1) * b.shape[-1], k)
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != b.shape[0]
        or offsets[-1] > n
        or any(e <= s for s, e in pairwise(offsets))
        or not all(t.is_contiguous() for t in (weight, a, b))
        or not isfinite(scaling)
    ):
        raise ValueError("Invalid packed FP8 merge factors, rank or offsets")
    if scaling == 0:
        return FP8Projection(weight, weight_scale)
    exact = weight.float() * weight_scale.float().repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    for i, (start, end) in enumerate(pairwise(offsets)):
        delta = torch.empty((end - start, k), dtype=torch.float32, device=weight.device)
        torch.mm(
            b[start:end, :rank].contiguous(), a[i * rank : (i + 1) * rank], out=delta
        )
        delta.mul_(scaling)
        exact[start:end].add_(delta)
        del delta
    # Match the independent checkpoint export: quantize with the scales that
    # will actually be stored, rather than quantizing with unrounded maxima.
    scales = (
        (
            exact.reshape(n // 128, 128, k // 128, 128)
            .abs()
            .amax((1, 3))
            .clamp_min(1e-12)
            / 448
        )
        .bfloat16()
        .float()
    )
    expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    rounded = (exact / expanded).clamp(-448, 448).to(torch.float8_e4m3fn)
    return FP8Projection(rounded.contiguous(), scales.contiguous())


class _ProjectionView:
    """Override weight operands without mutating the model or its Parameters."""

    def __init__(self, base, projection):
        self._base = base
        self.weight = projection.weight
        self.weight_scale_inv = projection.weight_scale_inv
        self._block_fp8_bf16_weight = None

    def __getattr__(self, name):
        return getattr(self._base, name)


class MergedFP8LoRABackend(MergedBF16LoRABackend):
    name = "merged_fp8"
    supports_dp_attention = False
    cache_size_field = "lora_merged_fp8_cache_size_mb"
    cache_policy_field = "lora_merged_fp8_cache_policy"

    def __init__(self, max_loras_per_batch, device, server_args, **kwargs):
        # Checkpoint-derived quantization is a declaration over raw CLI inputs.
        super().__init__(
            max_loras_per_batch, device, resolved_view(server_args), **kwargs
        )

    @staticmethod
    def _validate_server_args(server_args):
        server_args = resolved_view(server_args)
        requirements = {
            "dp_size=1": server_args.dp_size == 1,
            "pp_size=1": server_args.pp_size == 1,
            "cuda_graph_disabled": cuda_graph_fully_disabled(),
            "TF32_disabled": not torch.backends.cuda.matmul.allow_tf32,
            "deterministic_inference": server_args.enable_deterministic_inference,
            "quantization=fp8": server_args.quantization == "fp8",
            "speculation_none_or_EAGLE": server_args.speculative_algorithm
            in (None, "EAGLE"),
            "overlap_loading_disabled": not server_args.enable_lora_overlap_loading,
            "explicit_rank_1_to_64": server_args.max_lora_rank is not None
            and 0 < server_args.max_lora_rank <= 64,
        }
        missing = [name for name, satisfied in requirements.items() if not satisfied]
        if missing:
            raise ValueError("merged_fp8 unmet requirements: " + ", ".join(missing))

    def _request_token_lengths(self, forward_batch):
        if forward_batch.forward_mode.is_target_verify():
            width = forward_batch.spec_info.draft_token_num
            if not isinstance(width, int) or width <= 0:
                raise ValueError(
                    "merged_fp8 requires positive uniform EAGLE verify width"
                )
            return [width] * forward_batch.batch_size
        return super()._request_token_lengths(forward_batch)

    def _projection_bytes(self, layer, weight):
        size = super()._projection_bytes(layer, weight)
        if weight.dtype == torch.float8_e4m3fn:
            size += layer.base_layer.weight_scale_inv.numel() * 4
        return size

    def _merge_weight(self, layer, weight, a, b, rank, scale, offsets):
        if weight.dtype == torch.bfloat16:
            return merge_bf16_weight(weight, a, b, rank, scale, offsets)
        return merge_fp8_weight(
            weight, layer.base_layer.weight_scale_inv, a, b, rank, scale, offsets
        )

    def validate_lora_targets(self, base_model, target_modules):
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        allowed = {
            "gate_up_proj",
            "down_proj",
            "qkv_proj",
            "o_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
        }
        if target_modules - allowed:
            raise ValueError(
                f"merged_fp8 supports only dense targets {sorted(allowed)}"
            )
        for name, module in base_model.named_modules():
            if name.split(".")[-1] not in target_modules:
                continue
            if not isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                raise TypeError(f"merged_fp8 unsupported target: {name}")
            method, weight = module.quant_method, module.weight
            if (
                isinstance(method, UnquantizedLinearMethod)
                and weight.dtype == torch.bfloat16
            ):
                size = weight.numel() * weight.element_size()
            elif (
                isinstance(method, Fp8LinearMethod)
                and method.block_quant
                and method.weight_block_size == [128, 128]
                and not method.use_marlin
                and not method.use_mxfp8
                and not method.block_fp8_as_mxfp8
                and weight.dtype == torch.float8_e4m3fn
                and weight.is_contiguous()
                and not getattr(module, "aiter_bpreshuffled", False)
                and getattr(module, "_block_fp8_bf16_weight", None) is None
            ):
                n, k = weight.shape
                scales = module.weight_scale_inv
                if n % 128 or k % 128 or scales.shape != (n // 128, k // 128):
                    raise ValueError(f"merged_fp8 unaligned target: {name}")
                if not torch.equal(scales.float(), scales.bfloat16().float()):
                    raise ValueError(
                        f"merged_fp8 requires BF16 checkpoint block scales: {name}"
                    )
                size = weight.numel() + scales.numel() * 4
            else:
                raise ValueError(f"merged_fp8 unsupported quantization/layout: {name}")
            if self._cache_policy == "active":
                self._cache_transient_reserve = max(
                    self._cache_transient_reserve, min(self._weight_cache_limit, size)
                )

    @staticmethod
    def _linear(layer, x, bias, projection=None):
        if projection is None:
            return layer.base_layer.quant_method.apply(layer.base_layer, x, bias)
        if isinstance(projection, torch.Tensor):
            return F.linear(x, projection, bias)
        return layer.base_layer.quant_method.apply(
            _ProjectionView(layer.base_layer, projection), x, bias
        )

    def run_merged_linear(self, layer, x, bias):
        a, b = getattr(layer, "A_buffer", None), getattr(layer, "B_buffer", None)
        if a is None:
            a, b = layer.A_buffer_qkv, layer.B_buffer_qkv
        weight = layer.base_layer.weight
        present = getattr(layer, "merged_bf16_adapter_present", None)
        routing = self.merge_groups
        if (
            self.global_batch_info is not None
            and self.get_batch_info() is self.global_batch_info
        ):
            routing = self.global_merge_groups
        groups = [
            (slot, rank if present is None or present[slot] else 0, scale, rows)
            for slot, rank, scale, rows in routing
        ]
        if not any(rank for _, rank, _, _ in groups):
            return self._linear(layer, x, bias)
        offsets = layer.output_offset_cpu.tolist()
        if len(groups) == 1:
            slot, rank, scale, _ = groups[0]
            merged = self._merged_weight(
                layer, weight, a, b, slot, rank, scale, offsets
            )
            return self._linear(layer, x, bias, merged)
        output = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        for slot, rank, scale, rows in groups:
            merged = (
                self._merged_weight(layer, weight, a, b, slot, rank, scale, offsets)
                if rank
                else None
            )
            values = self._linear(layer, x.index_select(0, rows), bias, merged)
            output.index_copy_(0, rows, values)
        return output
