"""Independent FP64 attention oracle; never import or enable server dispatchers."""
import argparse
import hashlib
import json
import os
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main(root, rank, world):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    files=sorted(root.glob('*-rank*-layer*.pt'))
    assert files
    result=[]
    def stats(a,b):
        e=(a.double()-b.double()).abs()
        return {'count':e.numel(),'mean_abs':e.mean().item(),'max_abs':e.max().item(),
                'unequal':torch.count_nonzero(e).item()}
    for i,p in enumerate(files):
        if i%world!=rank:
            continue
        cpu=torch.load(p,map_location='cpu',weights_only=True)
        meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['k_pool_vs_source']['max_abs']==meta['v_pool_vs_source']['max_abs']==0
        q,k,v=(cpu[n].cuda().double() for n in ('q','k','v'))
        outputs=[]
        start=0
        for length in cpu['lengths']:
            qi=q[start:start+length].transpose(0,1)
            ki=k[start:start+length].transpose(0,1)
            vi=v[start:start+length].transpose(0,1)
            scores=(qi@ki.transpose(-1,-2))*cpu['scale']
            pos=torch.arange(length,device=q.device)
            scores.masked_fill_(pos[None,:]>pos[:,None],-float('inf'))
            outputs.append((scores.softmax(-1)@vi).transpose(0,1))
            start+=length
        gold=torch.cat(outputs).cpu()
        assert start==q.shape[0] and gold.isfinite().all()
        a=stats(cpu['candidate'],gold)
        b=stats(cpu['dense'],gold)
        c=stats(cpu['candidate'],cpu['dense'])
        assert c['count']==meta['candidate_vs_dense']['count']
        assert c['max_abs']==meta['candidate_vs_dense']['max_abs']
        row={'file':p.name,'raw_tensor_sha256':sha(p),'capture_metadata_sha256':sha(p.with_suffix('.json')),
             'label':meta['label'],'tp_rank':meta['tp_rank'],'layer_id':meta['layer_id'],
             'lengths':cpu['lengths'],'k_pool_exact':True,'v_pool_exact':True,
             'candidate_vs_fp64':a,'dense_vs_fp64':b,'candidate_vs_dense':c}
        result.append(row)
        print(json.dumps(row),flush=True)
        del q,k,v,gold,scores,outputs
    destination=root/f'oracle-rank{rank}.json'
    with destination.open('x') as f:
        json.dump({'oracle_dtype':'float64','worker_rank':rank,'worker_count':world,
                   'records':result,'source_sha256':sha(Path(__file__)),
                   'scope':'Attention-only error diagnosis on actual model tensors, not a model accuracy gate, speed or capacity result.'},f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,required=True)
    a=p.parse_args();main(a.root,a.rank,a.world)
