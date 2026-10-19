"""Recompute million-total natural-SSE measurements from immutable raw responses."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--rounds', type=int, choices=(2,), required=True)
a = p.parse_args()
R = Path(__file__).resolve().parent
S = R.parent/'qwen38-mol-speed-20261019'
O = S/'perf-long-yarn4-r26'
raw_rows = [json.loads(x) for x in (O/'requests.jsonl').read_text().splitlines()]
raw_groups = [json.loads(x) for x in (O/'groups.jsonl').read_text().splitlines()]
request_index = {(r['side'], r['label']): r for r in raw_rows}
group_index = {(r['side'], r['round'], r['cache']): r for r in raw_groups}
assert len(request_index) == len(raw_rows)
assert len(group_index) == len(raw_groups)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def near(x, y):
    assert math.isfinite(x) and math.isfinite(y)
    assert abs(x-y) <= 1e-8, (x,y)
configs = {}
for side in ('new','M','base'):
    path = O/f'{side}-get_server_info.json'
    cfg = json.loads(path.read_text())
    assert cfg['context_length'] == 1000000
    assert cfg['tp_size'] == 1 and cfg['disable_cuda_graph']
    assert cfg['enable_deterministic_inference']
    override = json.loads(cfg['json_model_override_args'])
    rope = override['text_config']['rope_parameters']
    assert rope['rope_type'] == 'yarn' and rope['factor'] == 4
    assert rope['original_max_position_embeddings'] == 262144
    configs[side] = {'file':path.name,'sha256':sha(path),'context_length':cfg['context_length'],
                     'model_path':cfg['model_path'],'tp_size':cfg['tp_size'],
                     'rope_override':override,'enable_lora':cfg['enable_lora']}
assert configs['new']['enable_lora'] and not configs['M']['enable_lora'] and not configs['base']['enable_lora']
assert len({json.dumps(c['rope_override'],sort_keys=True) for c in configs.values()}) == 1
rows=[]; outputs={}; prompt=None; payload=None
for repeat in range(a.rounds):
    for cache in ('cold','hot'):
        label = f'in999744-out256-c1-r{repeat}-{cache}-i0'
        for side in ('new','M','base'):
            path = O/f'{label}-{side}.json'
            data = json.loads(path.read_text()); req=data['request']; events=data['events']
            assert set(req) == ({'input_ids','stream','sampling_params','lora_path'} if side=='new' else {'input_ids','stream','sampling_params'})
            assert req['stream'] is True
            assert req['sampling_params'] == {'temperature':0,'ignore_eos':True,'max_new_tokens':256}
            assert len(req['input_ids']) == 999744
            normalized = {k:v for k,v in req.items() if k!='lora_path'}
            if prompt is None:
                prompt=req['input_ids']; payload=normalized
            else:
                assert req['input_ids'] == prompt and normalized == payload
            if side=='new': assert req['lora_path']=='perf_real'
            assert events and all(x['elapsed_s'] <= y['elapsed_s'] for x,y in zip(events,events[1:]))
            increasing=[]; prev=0
            for event in events:
                ids=event['data'].get('output_ids',[])
                assert len(ids)>=prev
                if len(ids)>prev:
                    increasing.append(event);prev=len(ids)
            assert len(increasing[0]['data']['output_ids']) == 1
            final=events[-1]['data']; ids=final['output_ids']; meta=final['meta_info']
            assert len(ids) == 256 and meta['prompt_tokens'] == 999744
            assert meta['finish_reason']['type'] == 'length'
            cached=meta.get('cached_tokens',0)
            assert (cached==0 if cache=='cold' else 0 < cached < 999744)
            first=increasing[0]['elapsed_s']; last=increasing[-1]['elapsed_s']
            tpot=(last-first)/255
            source=request_index[(side,label)]
            assert source['input_tokens']==999744 and source['output_tokens']==256
            assert source['concurrency']==1 and source['round']==repeat and source['cache']==cache
            assert source['output_ids']==ids and source['cached_tokens']==cached
            near(source['ttft_s'], first); near(source['tpot_s'], tpot)
            assert source['e2e_s'] >= events[-1]['elapsed_s']
            group=group_index[(side,repeat,cache)]
            assert group['input_tokens']==999744 and group['output_tokens']==256 and group['concurrency']==1
            assert group['elapsed_s'] >= source['e2e_s']
            near(group['output_throughput_tokens_s'],256/group['elapsed_s'])
            outputs[(side,repeat,cache)]=ids
            rows.append({'side':side,'round':repeat,'cache':cache,'raw_file':path.name,'sha256':sha(path),
                         'input_tokens':999744,'output_tokens':256,'cached_tokens':cached,
                         'first_sse_output_tokens':1,'sse_events':len(events),
                         'ttft_s':first,'tpot_s':tpot,'e2e_s':source['e2e_s'],
                         'group_elapsed_s':group['elapsed_s'],
                         'output_throughput_tokens_s':256/group['elapsed_s']})
for repeat in range(a.rounds):
    for cache in ('cold','hot'):
        assert outputs[('new',repeat,cache)]==outputs[('M',repeat,cache)]
assert len(rows) == a.rounds*6
points=[]
for side in ('new','M','base'):
    for cache in ('cold','hot'):
        rr=[r for r in rows if r['side']==side and r['cache']==cache]
        points.append({'side':side,'cache':cache,'requests':len(rr),
                       **{k:sum(r[k] for r in rr)/len(rr) for k in ('ttft_s','tpot_s','e2e_s','output_throughput_tokens_s')},
                       'cached_tokens_min':min(r['cached_tokens'] for r in rr),
                       'cached_tokens_max':max(r['cached_tokens'] for r in rr)})
if a.rounds==3:
    s=json.loads((O/'summary.json').read_text())
    assert s['request_count']==18 and s['natural_greedy_D_M_compared_requests']==6
    assert s['D_M_greedy_outputs_equal'] and len(s['points'])==6
    for point in points:
        q=next(q for q in s['points'] if q['side']==point['side'] and q['cache']==point['cache'])
        assert q['requests']==q['rounds']==3 and q['success_rate']==1.0
        for metric in ('ttft_s','tpot_s','e2e_s'):near(point[metric],q[metric]['mean'])
        near(point['output_throughput_tokens_s'],q['throughput_tokens_s_mean'])
result={'passed':True,'measured_rounds_audited':a.rounds,'formal_native_requests_audited':len(rows),
        'formal_native_requests_required':18,'complete_measured_matrix':a.rounds==3,
        'dynamic_merged_native_id_pairs_equal':a.rounds*2,'total_tokens_per_request':1000000,
        'warmups_included':False,'logprob_alignment_measured':False,'long_context_quality_accepted':False,
        'scope':'TP1 BF16 eager deterministic; identical fixed 999744-token prompt and 256 natural output tokens. D/M/base ran on separate GPUs while other experiments ran concurrently. These two rounds are partial evidence; full 18-request matrix requires three rounds. Native ID agreement and runtime completion are not full-prompt logprob or long-context task quality acceptance.',
        'client_provenance_sha256':sha(O/'client-provenance-r42.json'),
        'recovery_audit_sha256':sha(R/'long-client-timeout-recovery-audit-r43.json'),
        'audit_source_sha256':sha(Path(__file__)),
        'derived_from_r60_auditor_sha256':sha(R/'audit_million_measured_r60.py'),
        'configurations':configs,'points':points,'raw_request_audits':rows}
out=R/'million-two-measured-rounds-audit-r68.json'
assert not out.exists(), 'Do not overwrite evidence'
out.write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':True,'rounds':a.rounds,'native_payloads':len(rows),'points':points}))
