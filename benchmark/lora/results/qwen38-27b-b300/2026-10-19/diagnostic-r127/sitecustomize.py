"""Explicit, experiment-only import hook; preserve candidate model output."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

ROOT = os.environ.get('MOL_DCP_DIAGNOSTIC_ROOT_R127')

def install(module):
    import torch
    from sglang.srt.runtime_context import get_parallel
    original = module.TritonAttnBackend._forward_extend_dcp
    seen = set()
    layers = {}

    def wrapped(self, q, k, v, layer, batch, causal, cap, sinks):
        result = original(self, q, k, v, layer, batch, causal, cap, sinks)
        marker = Path(ROOT)/'enabled.json'
        if not marker.exists():
            return result
        label = json.loads(marker.read_text())['label']
        key = (label, layer.layer_id)
        chosen = layers.setdefault(label, set())
        if key in seen or (len(chosen) >= 2 and layer.layer_id not in chosen):
            return result
        if not bool((batch.extend_prefix_lens == 0).all().item()):
            return result
        seen.add(key)
        chosen.add(layer.layer_id)
        rank = get_parallel().attn_tp_rank
        group = get_parallel().dcp_group
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        k3 = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim).contiguous()
        v3 = v.view(-1, layer.tp_k_head_num, layer.v_head_dim).contiguous()
        assert layer.tp_k_head_num == 1
        lengths = batch.extend_seq_lens_cpu
        owned = batch.positions % self.dcp_size == self.dcp_rank
        write_ids = batch.out_cache_loc[owned] // self.dcp_size
        kb = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vb = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        kp = kb[write_ids]
        vp = vb[write_ids]
        counts = torch.tensor(lengths, device=q.device, dtype=torch.int32)
        ptr = torch.cat([torch.zeros(1,device=q.device,dtype=torch.int32),counts.cumsum(0)]).int()
        ids = torch.arange(k3.shape[0],device=q.device,dtype=torch.int64)
        dense = torch.empty_like(q3)
        self.extend_attention_fwd_unified(
            q3,dense,k3,v3,1.,1.,self.forward_metadata.qo_indptr,ptr,ids,
            batch.extend_prefix_lens,max(lengths),sm_scale=layer.scaling,is_causal=causal)
        golds = []
        start = 0
        for n in lengths:
            a = q3[start:start+n].double().transpose(0,1)
            b = k3[start:start+n].double().transpose(0,1)
            c = v3[start:start+n].double().transpose(0,1)
            scores = (a @ b.transpose(-1,-2))*layer.scaling
            pos = torch.arange(n,device=q.device)
            scores.masked_fill_(pos[None,:] > pos[:,None],-float('inf'))
            golds.append((scores.softmax(-1) @ c).transpose(0,1).float())
            start += n
        gold = torch.cat(golds)
        out = result.reshape_as(q3)
        def stats(a,b):
            delta = (a.float()-b.float()).abs()
            return {'mean_abs':delta.mean().item(),'max_abs':delta.max().item(),
                    'unequal':int(torch.count_nonzero(delta).item()),'count':delta.numel()}
        data = {'label':label,'tp_rank':rank,'dcp_rank':self.dcp_rank,
                'dcp_group_ranks':getattr(group,'ranks',None),'layer_id':layer.layer_id,
                'q_shape':list(q3.shape),'k_shape':list(k3.shape),
                'kv_pool_type':type(self.token_to_kv_pool).__name__,'lengths':lengths,
                'k_pool_vs_source':stats(kp,k3[owned]),'v_pool_vs_source':stats(vp,v3[owned]),
                'candidate_vs_dense':stats(out,dense),'candidate_vs_fp64':stats(out,gold),
                'dense_vs_fp64':stats(dense,gold),'candidate_finite':bool(out.isfinite().all().item()),
                'out_cache_loc_is_physical':batch.out_cache_loc_is_physical,
                'runtime_image':'sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878',
                'scope':'Real model Q/K/V, exact KV write read-back and attention-only FP64 oracle. Candidate output is returned unchanged; diagnostic timing is not a speed result.'}
        dest = Path(ROOT)/f'{label}-rank{rank}-layer{layer.layer_id}'
        with dest.with_suffix('.json').open('x') as f:
            json.dump(data,f,indent=2)
        torch.save({'q':q3.cpu(),'k':k3.cpu(),'v':v3.cpu(),'candidate':out.cpu(),
                    'dense':dense.cpu(),'oracle':gold.cpu(),'lengths':lengths,
                    'scale':layer.scaling},dest.with_suffix('.pt'))
        print('DCP_REAL_QKV_R127 '+json.dumps(data),flush=True)
        return result

    module.TritonAttnBackend._forward_extend_dcp = wrapped

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname != 'sglang.srt.layers.attention.triton_backend':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname,path)
        loader = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):
                return loader.create_module(spec)
            def exec_module(self,module):
                loader.exec_module(module)
                install(module)
        spec.loader = Loader()
        return spec

if ROOT:
    sys.meta_path.insert(0,Finder())
