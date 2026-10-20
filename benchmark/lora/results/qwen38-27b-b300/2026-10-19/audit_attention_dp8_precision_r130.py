"""Reconstruct original per-rank TP/attentionDP/DCP full-merge precision scores.

Read only: no SGLang imports, GPU requests, benchmark reruns or barrier mutations.
Intermediate *-scores.json and green summaries are checked against raw responses
and pre-forcing rank captures, never treated as the numerical source of truth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PHASES = ('prefill', 'prefill_last', 'decode')
IMAGE = 'sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
COMMIT = 'a589b336780051bbf26755cf1ea0aba56fbd029c'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def numerical_stats(pairs):
    assert pairs
    errors = []
    matches = 0
    for (a, ai), (b, bi) in pairs:
        assert math.isfinite(a) and math.isfinite(b), (a, b)
        assert type(ai) is int and type(bi) is int
        errors.append(abs(a-b))
        matches += ai == bi
    ordered = sorted(errors)
    mean = math.fsum(errors)/len(errors)
    p99 = ordered[math.ceil(.99*len(errors))-1]
    return {'count':len(errors), 'mean_abs_nat':mean, 'p99_abs_nat':p99,
            'max_abs_nat':ordered[-1], 'argmax_matches':matches,
            'argmax_match_fraction':matches/len(errors),
            'passes_logprob_gate':mean <= .01 and p99 <= .1}


def check_statistics(actual, recorded):
    for key, value in actual.items():
        if type(value) is float:
            assert math.isfinite(recorded[key])
            assert math.isclose(value, recorded[key], rel_tol=1e-12, abs_tol=1e-12), (key, value, recorded[key])
        else:
            assert recorded[key] == value, (key, value, recorded[key])
    assert actual['passes_logprob_gate'], actual


def native_scores(response, captures, case, continuation):
    assert len(continuation) >= 2 and response['output_ids'] == continuation
    rows = [json.loads(s) for s in captures.read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(len(continuation)))
    assert [r['token'] for r in rows] == continuation
    meta = response['meta_info']
    assert meta['prompt_tokens'] == len(case['input_ids'])
    assert meta['completion_tokens'] == len(continuation)
    assert meta['finish_reason']['type'] == 'length'
    lp, top = meta['input_token_logprobs'], meta['input_top_logprobs']
    assert len(lp) == len(top) == len(case['input_ids'])
    assert [row[1] for row in lp[1:]] == case['input_ids'][1:]
    scores = {'prefill':[(p[0], t[0][1]) for p,t in zip(lp[1:], top[1:])],
              'prefill_last':[(rows[0]['logprob'], rows[0]['argmax'])],
              'decode':[(r['logprob'],r['argmax']) for r in rows[1:]]}
    for values in scores.values():
        assert values and all(math.isfinite(v) and type(i) is int for v,i in values)
    return scores


def capture_scores(directory,label,response,case,continuation,tp,dp):
    assert tp>0 and dp>0 and tp%dp==0
    replicas=tp//dp
    paths=sorted(directory.glob(label+'.jsonl.rank-*'))
    assert len(paths)==replicas,(label,paths,replicas)
    ranks={int(f.name.rsplit('.rank-',1)[1]) for f in paths}
    assert len(ranks)==replicas and ranks<=set(range(tp))
    if dp==1:assert ranks==set(range(tp))
    scores=[native_scores(response,path,case,continuation) for path in paths]
    assert all(other==scores[0] for other in scores[1:])
    return scores[0],len(paths),ranks


def audit(root, top, label, case_count, rounds, tokens, expected_scores, legacy=False):
    tp = 4 if top in ('tp4','dpa4') else 8
    dp = tp if top.startswith('dpa') else 1
    dcp = 2 if top=='dcp8x2' else 1
    replicas=tp//dp
    rank_coverage={s:set() for s in ('D','M')}
    directories={side:root/f'{top}-{label}-{stage}-r130'
                 for side,stage in [('M','reference'),('D','dynamic')]}
    snapshots={side:inventory(p) for side,p in directories.items()}
    seal=read(directories['M']/'seal.json')
    assert set(snapshots['M']) == set(seal['files'])|{'seal.json'}
    assert all(snapshots['M'][name]==sha for name,sha in seal['files'].items())
    contracts={side:read(p/'contract.json') for side,p in directories.items()}
    assert contracts['D']==contracts['M']
    contract=contracts['M'];corpus=contract['corpus']
    assert len(corpus)==case_count and len({c['id'] for c in corpus})==case_count
    assert (contract['rounds'],contract['decode_tokens'],contract['batch_size'])==(rounds,tokens,case_count)
    assert contract['topology']==top+'-r130'
    corpus_file=root.parent/('qwen38-mol-experiment/phase1/corpus.json' if label=='full' else 'qwen38-mol-speed-20261019/corpus-mixed.json')
    assert corpus==read(corpus_file)[:case_count]
    configs={}
    for side,p in directories.items():
        provenance=read(p/'provenance.json')
        assert provenance['image']==IMAGE
        assert not legacy
        assert provenance['runtime_commit']==provenance['commit']==COMMIT
        if not legacy:assert provenance==read(root/'source-host-r130.json')
        cfg=read(p/'server-info.json');cfg=cfg.get('server_args',cfg);configs[side]=cfg
        assert cfg['tp_size']==tp and cfg['attn_dp_size']==dp and cfg['pp_size']==1
        assert cfg['dcp_size']==dcp
        if dcp>1:assert cfg['dcp_comm_backend']=='a2a'
        assert cfg['dtype'] in ('auto','bfloat16') and cfg['enable_deterministic_inference'] and cfg['disable_cuda_graph']
        assert cfg['quantization'] is None and cfg['enable_tf32_matmul'] is False
        model_path=Path(cfg['model_path']);assert model_path.is_relative_to('/work')
        model_config=read(root.parent/model_path.relative_to('/work')/'config.json')
        text_config=model_config.get('text_config',model_config)
        assert text_config.get('dtype',text_config.get('torch_dtype'))=='bfloat16'
        assert cfg['max_total_num_tokens']>0 and cfg['max_mamba_cache_size']==(32 if dp==8 else 256)
        assert cfg.get('lora_merged_bf16_cache_size_mb',0)==0
        assert bool(cfg['enable_lora'])==(side=='D')
    # Full configs are retained; explicitly check every numerical execution key
    # used by the client contract, rather than paths and LoRA-only settings.
    keys=('tp_size','dp_size','attn_dp_size','pp_size','dtype','kv_cache_dtype',
          'enable_tf32_matmul','context_length','chunked_prefill_size','max_running_requests',
          'attention_backend','prefill_attention_backend','decode_attention_backend',
          'linear_attn_backend','linear_attn_prefill_backend','linear_attn_decode_backend',
          'bf16_gemm_backend','enable_deterministic_inference','disable_cuda_graph',
          'json_model_override_args','max_mamba_cache_size','mamba_ssm_dtype',
          'mamba_radix_cache_strategy','disable_radix_cache','disable_chunked_prefix_cache',
          'page_size','disaggregation_mode','disaggregation_transfer_backend','dcp_size',
          'dcp_comm_backend','dcp_replicate_q_proj')
    assert all(configs['D'].get(k)==configs['M'].get(k) for k in keys)
    stored_cases={side:[json.loads(s) for s in (p/'cases.jsonl').read_text().splitlines()]
                  for side,p in directories.items()}
    cursor={side:0 for side in directories}
    pair_lists={phase:[] for phase in PHASES}
    noise={side:{phase:[] for phase in PHASES} for side in directories}
    first={};cached={side:{} for side in directories};capture_files=0
    for start in range(0,len(corpus),contract['batch_size']):
        group=corpus[start:start+contract['batch_size']]
        refs=read(directories['M']/f'references-{start}.json')
        assert read(directories['D']/f'references-{start}.json')==refs
        assert len(refs)==len(group)
        continuations=[r['output_ids'] for r in refs]
        assert all(len(c)==tokens for c in continuations)
        for repeat in range(rounds):
            for cache in ('cold','hot'):
                name=f'g{start}-r{repeat}-{cache}'
                responses={side:read(p/f'{name}-responses.json') for side,p in directories.items()}
                intermediate={side:read(p/f'{name}-scores.json') for side,p in directories.items()}
                assert all(len(responses[side])==len(intermediate[side])==len(group) for side in directories)
                for i,(case,continuation) in enumerate(zip(group,continuations)):
                    scores={}
                    for side,p in directories.items():
                        scores[side],ncap,ranks=capture_scores(p/'captures',f'{name}-i{i}',responses[side][i],case,continuation,tp,dp)
                        rank_coverage[side].update(ranks)
                        capture_files+=ncap
                        assert json.loads(json.dumps(scores[side]))==intermediate[side][i]
                        row=stored_cases[side][cursor[side]];cursor[side]+=1
                        assert (row['case'],row['round'],row['cache'])==(case['id'],repeat,cache)
                        hit=responses[side][i]['meta_info'].get('cached_tokens')
                        assert row['cached_tokens']==hit
                        bucket=f'{cache}:{hit}';cached[side][bucket]=cached[side].get(bucket,0)+1
                        for phase in PHASES:
                            target=scores['M'][phase] if side=='M' else intermediate['M'][i][phase]
                            check_statistics(numerical_stats(list(zip(scores[side][phase],target))),row[phase])
                            key=(side,case['id'],phase)
                            if key in first:noise[side][phase].extend(zip(first[key],scores[side][phase]))
                            else:first[key]=scores[side][phase]
                    for phase in PHASES:pair_lists[phase].extend(zip(scores['D'][phase],scores['M'][phase]))
    summaries={side:read(p/'summary.json') for side,p in directories.items()}
    count=case_count*rounds*2
    assert all(cursor[side]==len(stored_cases[side])==summaries[side]['case_pairs']==count for side in directories)
    recomputed={phase:numerical_stats(values) for phase,values in pair_lists.items()}
    assert {p:v['count'] for p,v in recomputed.items()}==expected_scores
    for phase,values in recomputed.items():check_statistics(values,summaries['D']['scores'][phase])
    recomputed_noise={side:{phase:numerical_stats(values) for phase,values in phases.items()} for side,phases in noise.items()}
    for side,phases in recomputed_noise.items():
        for phase,values in phases.items():check_statistics(values,summaries[side]['repeat_noise'][phase])
    assert all(inventory(p)==snapshots[side] for side,p in directories.items()),'source evidence changed during audit'
    if label=='full':assert all(v==set(range(tp)) for v in rank_coverage.values())
    return {'passed':True,'topology':top,'sampling_replicas_per_request':replicas,'rank_capture_coverage':{k:sorted(v) for k,v in rank_coverage.items()},'case_pairs':count,'scores':recomputed,'repeat_noise':recomputed_noise,
            'scored_native_responses_per_side':count,'rank_capture_files_reparsed':capture_files,
            'raw_evidence_files_rehashed':{s:len(v) for s,v in snapshots.items()},
            'reference_seal_sha256':digest(directories['M']/'seal.json'),
            'dynamic_evidence_manifest_sha256':hashlib.sha256(json.dumps(snapshots['D'],sort_keys=True).encode()).hexdigest(),
            'cached_tokens_distribution':cached,'inputs_stable_during_audit':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--topology',choices=['tp8','dcp8x2','dpa8','tp4','dpa4'],required=True);p.add_argument('--diagnostic-legacy-topology',action='store_true')
    args=p.parse_args();assert not args.output.exists()
    assert args.diagnostic_legacy_topology==(args.topology in ['tp4','dpa4'])
    results={label:audit(args.root,args.topology,label,n,r,t,counts,args.diagnostic_legacy_topology) for label,n,r,t,counts in [
        ('full',30,4,256,{'prefill':49744,'prefill_last':240,'decode':61200}),
        ('gdn',4,1,16,{'prefill':3380,'prefill_last':8,'decode':120})]}
    report={'passed':True,'topology':args.topology,'runtime_image':IMAGE,'runtime_commit':COMMIT,'runtime_source_manifest_sha256':digest(args.root/'source-host-r130.json'),'accuracy':results,
            'case_pairs':sum(v['case_pairs'] for v in results.values()),
            'score_count':sum(x['count'] for v in results.values() for x in v['scores'].values()),
            'audit_source_sha256':digest(Path(__file__)),
            'full8GPU_accepted':not args.diagnostic_legacy_topology,
            'scope':'Original full-prompt responses and each sampling replica pre-forcing capture re-parsed. Precision only; native performance, actual placement, KV residency and transport are separate gates. Legacy TP4/DP4 replay is auditor validation and provides no new8GPU acceptance.'}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
