"""Actual two-rank NCCL, production DCP backend and unified-kernel regression.

Uses a small in-memory KV pool and a collective adapter around torch.distributed;
it does not claim model, serving, performance or capacity acceptance.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import datetime
import faulthandler
import signal
from pathlib import Path
from types import SimpleNamespace


def run(output, original_kernel):
    import torch
    import torch.distributed as dist
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(local_rank).total_memory)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
    import sglang.kernels.ops.attention.extend_attention as kernels
    import sglang.srt.layers.attention.triton_backend as backendmod
    rank,world=dist.get_rank(),dist.get_world_size()
    assert world==2

    class CollectiveAdapter:
        world_size=world
        rank_in_group=rank
        def all_gather(self,tensor,dim=0):
            pieces=[torch.empty_like(tensor) for _ in range(world)]
            dist.all_gather(pieces,tensor.contiguous())
            return torch.cat(pieces,dim=dim)
        def all_reduce(self,tensor):
            dist.all_reduce(tensor)
            return tensor
    group=CollectiveAdapter()
    backendmod.get_parallel=lambda:SimpleNamespace(dcp_group=group)
    backend=object.__new__(backendmod.TritonAttnBackend)
    backend.dcp_size,backend.dcp_rank=world,rank
    backend.device=torch.device('cuda',local_rank)
    backend.extend_attention_fwd=kernels.extend_attention_fwd
    backend.extend_attention_fwd_unified=kernels.extend_attention_fwd_unified
    layer=SimpleNamespace(tp_q_head_num=3,tp_k_head_num=1,qk_head_dim=256,
                          v_head_dim=256,layer_id=0,scaling=1/16,
                          k_scale=None,v_scale=None,xai_temperature_len=-1,
                          sliding_window_size=None)
    generator=torch.Generator(device=backend.device).manual_seed(116)
    cases=[]
    old_nonzero=0

    def prepare(lengths):
        qs=[];ks=[];vs=[];local_keys=[];local_values=[]
        req_table=torch.zeros((len(lengths),max(lengths)),dtype=torch.int32,device=backend.device)
        offset=0
        for i,n in enumerate(lengths):
            q=torch.randn(n,6,256,device=backend.device,dtype=torch.bfloat16,generator=generator)
            k=torch.randn(n,1,256,device=backend.device,dtype=torch.bfloat16,generator=generator)
            v=torch.randn(n,1,256,device=backend.device,dtype=torch.bfloat16,generator=generator)
            qs.append(q);ks.append(k);vs.append(v)
            slots=(n+1)//2
            pk=torch.zeros(slots,1,256,device=backend.device,dtype=torch.bfloat16)
            pv=torch.zeros_like(pk)
            striped_k=k[rank::2];striped_v=v[rank::2]
            pk[:len(striped_k)]=striped_k;pv[:len(striped_v)]=striped_v
            local_keys.append(pk);local_values.append(pv)
            req_table[i,:n]=torch.arange(offset*2,offset*2+n,device=backend.device,dtype=torch.int32)
            offset+=slots
        backend.req_to_token=req_table
        key_buffer=torch.cat(local_keys);value_buffer=torch.cat(local_values)
        backend.token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _:key_buffer,
                                                 get_value_buffer=lambda _:value_buffer)
        return qs,ks,vs

    def forward(qs,ks,vs,prefixes,ends,deterministic):
        counts=[end-prefix for prefix,end in zip(prefixes,ends)]
        reqs=torch.arange(len(qs),device=backend.device,dtype=torch.int32)
        lens=torch.tensor(ends if deterministic else prefixes,device=backend.device,dtype=torch.int32)
        ptr=torch.zeros(len(qs)+1,device=backend.device,dtype=torch.int32)
        ptr,indices,_=backend._dcp_kv_indices(reqs,lens,ptr)
        qi=torch.tensor([0]+counts,device=backend.device,dtype=torch.int32).cumsum(0).int()
        backend.forward_metadata=SimpleNamespace(qo_indptr=qi,kv_indptr=ptr,kv_indices=indices,
                                                 max_extend_len=max(counts),custom_mask=None)
        batch=SimpleNamespace(extend_prefix_lens=torch.tensor(prefixes,device=backend.device,dtype=torch.int32))
        q=torch.cat([x[p:e,rank*3:(rank+1)*3] for x,p,e in zip(qs,prefixes,ends)]).contiguous()
        k=torch.cat([x[p:e] for x,p,e in zip(ks,prefixes,ends)]).contiguous()
        v=torch.cat([x[p:e] for x,p,e in zip(vs,prefixes,ends)]).contiguous()
        backend.enable_deterministic=deterministic
        return backend._forward_extend_dcp(q,k,v,layer,batch,True,0.0,None).reshape(-1,3,256)

    for length in [1,127,128,129,874,1378,1630]:
        qs,ks,vs=prepare([length])
        whole=forward(qs,ks,vs,[0],[length],True)
        old_whole=forward(qs,ks,vs,[0],[length],False)
        print(json.dumps({'rank':rank,'phase':'whole','length':length}),flush=True)
        q64=qs[0][:,rank*3:(rank+1)*3].double().transpose(0,1)
        logits=(q64@ks[0].double().transpose(0,1).transpose(-1,-2))/16
        positions=torch.arange(length,device=backend.device)
        logits.masked_fill_(positions[None,:]>positions[:,None],-float('inf'))
        reference=(logits.softmax(-1)@vs[0].double().transpose(0,1)).transpose(0,1).float()
        oracle_error=(whole.float()-reference).abs()
        assert torch.isfinite(whole).all()
        oracle_pass=torch.allclose(whole.float(),reference,atol=.01,rtol=.01)
        cases.append({'kind':'FP64_oracle','length':length,'mean_abs':oracle_error.mean().item(),
                      'max_abs':oracle_error.max().item(),'passed':oracle_pass})
        del q64,logits,reference,oracle_error
        for cut in [1,2,63,127,128,129,768,1408]:
            if cut>=length:continue
            chunks=torch.cat([forward(qs,ks,vs,[0],[cut],True),
                              forward(qs,ks,vs,[cut],[length],True)])
            # The original path returns locally if prefix KV is empty. At
            # cut=1 one rank skips NCCL and the other does not (R116 stack
            # capture). Keep candidate cut=1; do not call that broken control.
            new_error=(whole.float()-chunks.float()).abs().max().item()
            old_error=None
            if cut>=world:
                old_chunks=torch.cat([forward(qs,ks,vs,[0],[cut],False),
                                      forward(qs,ks,vs,[cut],[length],False)])
                old_error=(old_whole.float()-old_chunks.float()).abs().max().item()
                old_nonzero+=old_error>0
            row={'kind':'chunk_invariance','length':length,'cut':cut,
                 'candidate_max_abs':new_error,'old_path_max_abs':old_error,
                 'old_control_skipped_empty_prefix_shard':cut<world,
                 'passed':new_error==0}
            cases.append(row)
            print(json.dumps({'rank':rank,**row}),flush=True)
        torch.cuda.empty_cache()
    # A batch-max grid also launches inactive query blocks for the short request.
    qs,ks,vs=prepare([1,129])
    batched=forward(qs,ks,vs,[0,0],[1,129],True)
    assert torch.isfinite(batched).all()
    cases.append({'kind':'ragged_1_129','finite':True,'passed':True})

    # Default DCP_SIZE=1 path must remain bit-identical to the original kernel.
    spec=importlib.util.spec_from_file_location('original_extend_attention_r117',original_kernel)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    for prefix,length in [(0,1),(0,256),(127,129),(768,106),(1408,222)]:
        n=prefix+length
        q=torch.randn(length,6,256,device=backend.device,dtype=torch.bfloat16,generator=generator)
        k=torch.randn(n,1,256,device=backend.device,dtype=torch.bfloat16,generator=generator)
        v=torch.randn_like(k)
        a=torch.empty_like(q);b=torch.empty_like(q)
        args=(k,v,1.,1.,torch.tensor([0,length],device=backend.device,dtype=torch.int32),
              torch.tensor([0,n],device=backend.device,dtype=torch.int32),
              torch.arange(n,device=backend.device,dtype=torch.int64),
              torch.tensor([prefix],device=backend.device,dtype=torch.int32),length)
        kernels.extend_attention_fwd_unified(q,a,*args,sm_scale=1/16,is_causal=True)
        original.extend_attention_fwd_unified(q,b,*args,sm_scale=1/16,is_causal=True)
        error=(a.float()-b.float()).abs().max().item()
        cases.append({'kind':'non_DCP_original_parity','prefix':prefix,'extend':length,
                      'max_abs':error,'passed':error==0})
    torch.cuda.synchronize()
    result={'rank':rank,'world_size':world,'passed':all(c['passed'] for c in cases) and old_nonzero>0,
            'old_nonzero_chunk_cases':old_nonzero,'cases':cases,
            'peak_pytorch_allocated_bytes':torch.cuda.max_memory_allocated(),
            'kernel_sha256':hashlib.sha256(Path(inspect.getfile(kernels)).read_bytes()).hexdigest(),
            'backend_sha256':hashlib.sha256(Path(inspect.getfile(backendmod)).read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    gathered=[None]*world;dist.all_gather_object(gathered,result)
    if rank==0:
        assert not output.exists()
        output.write_text(json.dumps({'passed':all(x['passed'] for x in gathered),'ranks':gathered,
            'full_goal_complete':False,'scope':'Two actual NCCL ranks exercise production backend, real sharded-KV index kernel, query/LSE all-gathers and output all-reduce. FP64 oracle, whole-vs-chunk and unmodified default-kernel parity. Not full-model D/M, mixed serving, performance or KV-capacity acceptance.'},indent=2))
        print(json.dumps({'passed':all(x['passed'] for x in gathered),'cases_per_rank':len(cases),'old_nonzero_per_rank':[x['old_nonzero_chunk_cases'] for x in gathered]}),flush=True)
    dist.destroy_process_group()
    return all(x['passed'] for x in gathered)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original-kernel',type=Path,required=True);a=p.parse_args()
    raise SystemExit(0 if run(a.output,a.original_kernel) else 1)
