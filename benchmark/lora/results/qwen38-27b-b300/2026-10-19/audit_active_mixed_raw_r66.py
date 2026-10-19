"""Read-only R55 mixed routing audit, rebuilding scores from raw captures."""
import argparse
import json
import math
import random
from pathlib import Path
from audit_active_raw_r65 import read, digest, inventory, numerical_stats, check_statistics


def main(root, output):
    assert not output.exists()
    directory=root/'active-mixed-r56';captures=root/'active-mixed-captures-r56'
    before={p.name:inventory(p) for p in [directory,captures]}
    summary=read(directory/'summary.json');refs=read(directory/'references.json')
    cfg=read(directory/'server-info.json');cfg=cfg.get('server_args',cfg)
    assert cfg['tp_size']==cfg['attn_dp_size']==1 and cfg['enable_lora']
    assert cfg['max_total_num_tokens']==65536 and cfg['max_mamba_cache_size']==256
    assert cfg['lora_merged_bf16_cache_policy']=='active' and cfg['lora_merged_bf16_cache_size_mb']==36864
    assert len(refs)==summary['isolated_requests']==16
    assert len({ref['id'] for ref in refs})==16
    assert {ref['adapter'] for ref in refs}=={None,'real','zero','gdn'}
    assert all(sum(ref['adapter']==adapter for ref in refs)==4 for adapter in (None,'real','zero','gdn'))
    assert summary['adapters_including_base']==4 and summary['mixed_requests']==len(summary['cases'])==48

    def native(response,label,ref):
        cont=ref['continuation'];assert len(cont)==16 and response['output_ids']==cont
        meta=response['meta_info'];assert meta['prompt_tokens']==len(ref['input_ids']) and meta['completion_tokens']==16
        assert meta['finish_reason']['type']=='length'
        paths=list(captures.glob(label+'.jsonl.rank-*'));assert len(paths)==1 and paths[0].name.endswith('.rank-0')
        rows=[json.loads(s) for s in paths[0].read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(16))
        assert [r['token'] for r in rows]==cont
        values=[(r['logprob'],r['argmax']) for r in rows]
        assert all(math.isfinite(v) and type(i) is int for v,i in values)
        return values

    isolated=[]
    for ref in refs:
        greedy=read(directory/(ref['id']+'-greedy.json'))
        assert len(greedy['output_ids'])==16 and greedy['output_ids']==ref['continuation']
        assert greedy['meta_info']['prompt_tokens']==len(ref['input_ids'])
        label=ref['id']+'-isolated'
        scores=native(read(directory/(label+'.json')),label,ref)
        assert [list(v) for v in scores]==ref['scores']
        isolated.append(scores)
    pairs={'prefill_last':[],'decode':[]};cursor=0
    rng=random.Random(summary['seed'])
    for repeat in range(3):
        order=list(range(16));rng.shuffle(order)
        responses=read(directory/f'mixed-r{repeat}.json');assert len(responses)==16
        for index,response in zip(order,responses):
            ref=refs[index];label=f'r{repeat}-'+ref['id'];actual=native(response,label,ref)
            saved=summary['cases'][cursor];cursor+=1
            assert (saved['id'],saved['round'])==(ref['id'],repeat)
            assert saved['cached_tokens']==response['meta_info'].get('cached_tokens')
            for phase,start,end in [('prefill_last',0,1),('decode',1,16)]:
                values=list(zip(actual[start:end],isolated[index][start:end]));pairs[phase].extend(values)
                check_statistics(numerical_stats(values),saved[phase])
    scores={phase:numerical_stats(values) for phase,values in pairs.items()}
    assert scores['prefill_last']['count']==48 and scores['decode']['count']==720
    for phase,stats in scores.items():check_statistics(stats,summary['scores'][phase])
    assert all(inventory(p)==before[p.name] for p in [directory,captures])
    result={'passed':True,'mixed_requests':48,'isolated_scored_requests':16,'natural_reference_requests':16,
            'raw_rank_capture_files_reparsed':64,'scores':scores,'raw_evidence_files_rehashed':sum(map(len,before.values())),
            'summary_sha256':digest(directory/'summary.json'),'audit_source_sha256':digest(Path(__file__)),
            'score_auditor_dependency_sha256':digest(Path(__file__).with_name('audit_active_raw_r65.py')),
            'scope':'R55 TP1 active-cache mixed routing versus per-identity isolated references. Four identities (base/real/zero/GDN), three shuffled batches. Original pre-forcing scores only; not independent D/M precision, full-prefill or real prefix-restore acceptance.'}
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
