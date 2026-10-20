"""Preserve failed DCP mixed numerical gates after fully reparsing raw captures."""
import argparse
import json
import math
import random
from pathlib import Path
from audit_active_raw_r65 import read, digest, inventory, numerical_stats


def main(root, output, top, legacy=False):
    tp=4 if legacy else 8;dp=tp if top.startswith('dpa') else 1;replicas=tp//dp
    suffix='r118'
    original=read(root/'source-topology-audits-r102.json')
    for name in ('audit_topology_mixed_r101.py','audit_active_raw_r65.py'):
        assert digest(root/name)==original['source_files'][name], name
    failures=[]
    def compare_saved(actual, recorded, label):
        for key, value in actual.items():
            if type(value) is float:
                assert math.isfinite(recorded[key]) and math.isclose(value,recorded[key],rel_tol=1e-12,abs_tol=1e-12)
            else:
                assert recorded[key]==value,(label,key,value,recorded[key])
        if not actual['passes_logprob_gate']:
            failures.append({'label':label,'statistics':actual})
    capture_count=0;rank_coverage=set()
    assert not output.exists()
    directory=root/f'{top}-early-mixed-{suffix}';captures=root/f'{top}-early-mixed-captures-{suffix}'
    before={p.name:inventory(p) for p in [directory,captures]}
    summary=read(directory/'summary.json');refs=read(directory/'references.json')
    cfg=read(directory/'server-info.json');cfg=cfg.get('server_args',cfg)
    assert cfg['tp_size']==tp and cfg['attn_dp_size']==dp and cfg['enable_lora']
    assert cfg['dcp_size']==(2 if top=='dcp8x2' else 1)
    assert cfg['max_total_num_tokens']>0 and cfg['max_mamba_cache_size']==(32 if dp==8 else 256)
    assert len(refs)==summary['isolated_requests']==16
    assert len({ref['id'] for ref in refs})==16
    assert {ref['adapter'] for ref in refs}=={None,'real','zero','gdn'}
    assert all(sum(ref['adapter']==adapter for ref in refs)==4 for adapter in (None,'real','zero','gdn'))
    assert summary['adapters_including_base']==4 and summary['mixed_requests']==len(summary['cases'])==48

    def native(response,label,ref):
        nonlocal capture_count
        cont=ref['continuation'];assert len(cont)==16 and response['output_ids']==cont
        meta=response['meta_info'];assert meta['prompt_tokens']==len(ref['input_ids']) and meta['completion_tokens']==16
        assert meta['finish_reason']['type']=='length'
        paths=sorted(captures.glob(label+'.jsonl.rank-*'));assert len(paths)==replicas
        ranks={int(p.name.rsplit('.rank-',1)[1]) for p in paths};assert len(ranks)==replicas and ranks<=set(range(tp))
        if dp==1:assert ranks==set(range(tp))
        rank_coverage.update(ranks);capture_count+=len(paths)
        copies=[[json.loads(s) for s in p.read_text().splitlines()] for p in paths]
        assert all(v==copies[0] for v in copies[1:]);rows=copies[0]
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
                compare_saved(numerical_stats(values),saved[phase],label+":"+phase)
    scores={phase:numerical_stats(values) for phase,values in pairs.items()}
    assert scores['prefill_last']['count']==48 and scores['decode']['count']==720
    for phase,stats in scores.items():compare_saved(stats,summary['scores'][phase],'aggregate:'+phase)
    assert all(inventory(p)==before[p.name] for p in [directory,captures])
    assert rank_coverage==set(range(tp))
    result={'passed':not failures,'failed_numerical_gates':failures,'full_goal_complete':False,'root_cause_proven':False,'original_auditor_sha256':digest(root/'audit_topology_mixed_r101.py'),'topology':top,'sampling_rank_coverage':sorted(rank_coverage),'full8GPU_accepted':not legacy and not failures,'mixed_requests':48,'isolated_scored_requests':16,'natural_reference_requests':16,
            'raw_rank_capture_files_reparsed':capture_count,'scores':scores,'raw_evidence_files_rehashed':sum(map(len,before.values())),
            'summary_sha256':digest(directory/'summary.json'),'audit_source_sha256':digest(Path(__file__)),
            'score_auditor_dependency_sha256':digest(Path(__file__).with_name('audit_active_raw_r65.py')),
            'scope':'Candidate R116 early serving regression, new R118 namespace. Per-rank mixed routing versus isolated references; fouridentities(base/real/zero/GDN),3shuffledbatches48requests/768scores. TP sampling replicas agree; attentionDP capture rank membership checked without assuming requests share the same DP rank. Not independent D/M, full-prefill, actual prefix-restoration or KV residency acceptance; legacy4GPU replay is parser validation only.'}
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--topology',choices=['tp8','dcp8x2','dpa8','tp4','dpa4'],required=True);p.add_argument('--diagnostic-legacy-topology',action='store_true')
    a=p.parse_args();assert a.diagnostic_legacy_topology==(a.topology in ['tp4','dpa4'])
    result=main(a.root,a.output,a.topology,a.diagnostic_legacy_topology)
    raise SystemExit(0 if result['passed'] else 1)
