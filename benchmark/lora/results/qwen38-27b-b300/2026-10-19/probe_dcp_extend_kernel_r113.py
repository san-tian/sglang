"""One-GPU attention arithmetic diagnostic; no distributed transport acceptance."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def run():
    import torch
    import sglang.kernels.ops.attention.extend_attention as kernels

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    device, dtype = 'cuda', torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(113)
    cases = []
    for prefix in (0, 127, 128, 129, 768, 1408):
        for length in (1, 128, 256):
            # TP8 local3 query heads; a DCP2 group gathers6 heads over one
            # replicated KV head. Prefix positions are striped over2 shards.
            q = torch.randn(length, 6, 256, device=device, dtype=dtype, generator=generator)
            k = torch.randn(prefix+length, 1, 256, device=device, dtype=dtype, generator=generator)
            v = torch.randn(prefix+length, 1, 256, device=device, dtype=dtype, generator=generator)
            qi = torch.tensor([0, length], device=device, dtype=torch.int32)
            empty_ptr = torch.zeros(2, device=device, dtype=torch.int32)
            empty_indices = torch.empty(0, device=device, dtype=torch.int64)
            current = torch.empty(length, 6, 256, device=device, dtype=torch.float32)
            current_lse = torch.empty(length, 6, device=device, dtype=torch.float32)
            kernels.extend_attention_fwd(
                q, k[prefix:].contiguous(), v[prefix:].contiguous(), current,
                k, v, qi, empty_ptr, empty_indices, None, True, None, length,
                1.0, 1.0, sm_scale=1/16, lse_extend=current_lse, skip_prefix=True,
            )
            partials, lses = [], []
            for rank in range(2):
                pk, pv = k[rank:prefix:2].contiguous(), v[rank:prefix:2].contiguous()
                out = torch.zeros_like(current)
                lse = torch.full_like(current_lse, -float('inf'))
                if pk.shape[0]:
                    indptr = torch.tensor([0, pk.shape[0]], device=device, dtype=torch.int32)
                    indices = torch.arange(pk.shape[0], device=device, dtype=torch.int64)
                    kernels.extend_attention_fwd(
                        q, k[:0], v[:0], out, pk, pv, qi, indptr, indices,
                        None, False, None, length, 1.0, 1.0, sm_scale=1/16,
                        lse_extend=lse, skip_extend=True,
                    )
                partials.append(out)
                lses.append(lse)
            # Reproduce the existing natural-log merge on one GPU. There is
            # no NCCL or rank transport in this arithmetic diagnostic.
            global_prefix_lse = torch.logsumexp(torch.stack(lses), dim=0)
            weights = torch.exp(torch.stack(lses)-global_prefix_lse).nan_to_num(nan=0.0)
            prefix_out = (torch.stack(partials)*weights[..., None]).sum(0)
            total_lse = torch.logaddexp(global_prefix_lse, current_lse)
            dcp = (prefix_out*torch.exp(global_prefix_lse-total_lse).nan_to_num(nan=0.0)[..., None]
                   + current*torch.exp(current_lse-total_lse)[..., None]).to(dtype)
            unified = torch.empty_like(q)
            kernels.extend_attention_fwd_unified(
                q, unified, k, v, 1.0, 1.0, qi,
                torch.tensor([0, prefix+length], device=device, dtype=torch.int32),
                torch.arange(prefix+length, device=device, dtype=torch.int64),
                torch.tensor([prefix], device=device, dtype=torch.int32), length,
                sm_scale=1/16, is_causal=True,
            )
            q64, k64, v64 = q.double().transpose(0, 1), k.double().transpose(0, 1), v.double().transpose(0, 1)
            logits = torch.matmul(q64, k64.transpose(-1, -2))/16
            mask = torch.arange(prefix+length, device=device)[None, :] <= prefix+torch.arange(length, device=device)[:, None]
            logits = logits.masked_fill(~mask, -float('inf'))
            reference = torch.matmul(logits.softmax(-1), v64).transpose(0, 1).float()
            finite = all(torch.isfinite(t).all().item() for t in (dcp, unified, reference))
            assert finite
            row = {'prefix_tokens': prefix, 'extend_tokens': length, 'gathered_query_heads': 6, 'kv_heads': 1, 'head_dim': 256}
            for label, tensor in [('dcp_vs_fp64', dcp.float()), ('unified_vs_fp64', unified.float())]:
                error = (tensor-reference).abs()
                row[label] = {'mean_abs': error.mean().item(), 'max_abs': error.max().item(),
                              'within_atol_rtol_0p01': torch.allclose(tensor, reference, atol=.01, rtol=.01)}
            row['dcp_vs_unified_max_abs'] = (dcp.float()-unified.float()).abs().max().item()
            cases.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    source = Path(inspect.getfile(kernels))
    return {'passed': all(c[k]['within_atol_rtol_0p01'] for c in cases for k in ('dcp_vs_fp64','unified_vs_fp64')),
            'scope': 'Actual immutable-image Triton prefix/current and unified extend kernels versus FP64 masked attention, on one GPU. DCP LSE combination is emulated locally; this is not a distributed-collective, full-model, D/M, latency, or KV-residency acceptance.',
            'cases': cases, 'kernel_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'peak_pytorch_allocated_bytes': torch.cuda.max_memory_allocated(), 'full_goal_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = run()
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    raise SystemExit(0 if result['passed'] else 1)
