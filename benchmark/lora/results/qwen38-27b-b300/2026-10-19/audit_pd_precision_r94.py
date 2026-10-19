"""Read-only reconstruction of original PD score records, not green flags.

Default contract is the complete queued TP4-per-role PD matrix. Reference-only
mode can exercise the parser on an existing smaller independent-M diagnostic;
its report explicitly does not accept D/M or full4P4D accuracy.
"""
import argparse,copy,json,math
from pathlib import Path
from audit_active_raw_r65 import read,digest,inventory,numerical_stats,check_statistics
PHASES=('prefill','prefill_last','decode')
IMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
BASE='/work/models/Qwen3.8-27B'
MERGES='/work/qwen38-mol-development-20261019/merged/'


def original_index(folder):
    indexed={};rooms=set()
    for path in sorted(folder.glob('pd-*-prefill.json')):
        pf=read(path);df=read(path.with_name(path.name.replace('-prefill.json','-decode.json')))
        n=len(pf['requests']);assert n==len(pf['responses'])==len(df['requests'])==len(df['responses'])
        for preq,dreq,pres,dres in zip(pf['requests'],df['requests'],pf['responses'],df['responses']):
            room=preq['bootstrap_room'];assert type(room) is int and 0<=room<2**63
            assert room==dreq['bootstrap_room'] and room not in rooms;rooms.add(room)
            if 'custom_logit_processor' not in preq:continue
            capture=Path(preq['sampling_params']['custom_params']['capture_file'])
            assert capture.parent.name=='captures' and capture.parent.parent.name==folder.name
            assert capture.name not in indexed
            indexed[capture.name]=(preq,dreq,pres,dres)
    assert indexed
    return indexed


def request_scores(folder,index,label,i,case,continuation,combined,replicas,dynamic):
    key=f'{label}-i{i}.jsonl.prefill';preq,dreq,pres,dres=index[key]
    nt=len(continuation)
    assert preq['input_ids']==dreq['input_ids']==case['input_ids']
    assert preq['bootstrap_host']==dreq['bootstrap_host']=='127.0.0.1'
    assert preq['bootstrap_port']==dreq['bootstrap_port']
    assert preq['bootstrap_room']==dreq['bootstrap_room']
    assert preq['custom_logit_processor']==dreq['custom_logit_processor']
    if dynamic:assert preq['lora_path']==dreq['lora_path']=='frozen-A'
    else:assert preq.get('lora_path') is None and dreq.get('lora_path') is None
    for role,req,start in [('prefill',preq,0),('decode',dreq,1)]:
        assert req['return_logprob'] is True and req['logprob_start_len']==0 and req['top_logprobs_num']==1
        params=req['sampling_params'];assert params['temperature']==0 and params['max_new_tokens']==nt and params['ignore_eos'] is True
        cp=params['custom_params'];assert cp['continuation']==continuation and cp['capture_start_step']==start
        assert Path(cp['capture_file']).name==f'{label}-i{i}.jsonl.{role}'
    assert pres['output_ids']==continuation[:1] and dres['output_ids']==continuation
    assert pres['meta_info']['completion_tokens']==1
    assert dres['meta_info']['completion_tokens']==nt and dres['meta_info']['finish_reason']=={'type':'length','length':nt}
    merged=copy.deepcopy(dres)
    for k,v in pres['meta_info'].items():
        if k.startswith('input_'):merged['meta_info'][k]=v
    assert merged==combined
    captures=[]
    for role,steps in [('prefill',[0]),('decode',list(range(1,nt)))]:
        files=sorted((folder/'captures').glob(f'{label}-i{i}.jsonl.{role}.rank-*'))
        assert len(files)==replicas and {int(f.name.rsplit('-',1)[1]) for f in files}==set(range(replicas))
        ranks=[[json.loads(s) for s in f.read_text().splitlines()] for f in files]
        assert all(r==ranks[0] for r in ranks)
        assert [r['step'] for r in ranks[0]]==steps
        assert [r['token'] for r in ranks[0]]==[continuation[s] for s in steps]
        captures.extend(ranks[0])
    meta=pres['meta_info'];assert meta['prompt_tokens']==len(case['input_ids'])
    lp,top=meta['input_token_logprobs'],meta['input_top_logprobs']
    assert len(lp)==len(top)==len(case['input_ids'])
    assert [p[1] for p in lp[1:]]==case['input_ids'][1:]
    result={'prefill':[(p[0],t[0][1]) for p,t in zip(lp[1:],top[1:])],
            'prefill_last':[(captures[0]['logprob'],captures[0]['argmax'])],
            'decode':[(r['logprob'],r['argmax']) for r in captures[1:]]}
    for scores in result.values():
        assert scores and all(math.isfinite(s) and type(t) is int for s,t in scores)
    return result


def audit(root,label,count,rounds,tokens,tp,source,prefix='pd4-r33',suffix='r30',reference_only=False):
    sides={'M':'reference'} if reference_only else {'M':'reference','D':'dynamic'}
    folders={s:root/f'{prefix}-{label}-{stage}-{suffix}' for s,stage in sides.items()}
    before={s:inventory(p) for s,p in folders.items()}
    seal=read(folders['M']/'seal.json')['files']
    assert before['M']==dict(seal,**{'seal.json':digest(folders['M']/'seal.json')})
    contract=read(folders['M']/'contract.json');corpus=contract['corpus']
    assert (contract['topology'],contract['rounds'],contract['decode_tokens'],contract['batch_size'])==(f'{prefix}-{suffix}',rounds,tokens,count)
    assert len(corpus)==len({c['id'] for c in corpus})==count
    manifest=read(root/source);assert manifest['runtime_image']==IMAGE
    config={}
    for side,folder in folders.items():
        assert read(folder/'contract.json')==contract and read(folder/'provenance.json')==manifest
        for role,filename,gpu in [('prefill','prefill-server-info.json',0),('decode','server-info.json',tp)]:
            info=read(folder/filename);cfg=info.get('server_args',info);config[side,role]=cfg
            assert cfg['tp_size']==tp and cfg['attn_dp_size']==cfg['pp_size']==1
            assert cfg['base_gpu_id']==gpu and cfg['disaggregation_mode']==role
            assert cfg['disaggregation_transfer_backend']=='nixl' and cfg['disaggregation_enable_kv_checksum']
            assert cfg['mamba_radix_cache_strategy']=='extra_buffer'
            assert cfg['quantization'] is None and cfg['enable_tf32_matmul'] is False
            assert cfg['enable_deterministic_inference'] and cfg['disable_cuda_graph']
            assert bool(cfg['enable_lora'])==(side=='D')
            adapter='real_r16' if label=='full' else 'gdn_fused'
            assert cfg['model_path']==(BASE if side=='D' else MERGES+adapter)
        model=read(folder/'model-info.json');assert model['model_path']==config[side,'decode']['model_path']
    keys=('dtype','context_length','chunked_prefill_size','max_running_requests','attention_backend',
          'prefill_attention_backend','decode_attention_backend','linear_attn_backend',
          'linear_attn_prefill_backend','linear_attn_decode_backend','bf16_gemm_backend','kv_cache_dtype',
          'max_mamba_cache_size','mamba_ssm_dtype','json_model_override_args','page_size',
          'disable_radix_cache','disable_chunked_prefix_cache','dcp_size','dcp_comm_backend')
    for role in ['prefill','decode']:
        if not reference_only:assert all(config['M',role].get(k)==config['D',role].get(k) for k in keys)
    indices={s:original_index(p) for s,p in folders.items()}
    cases={s:[json.loads(x) for x in (p/'cases.jsonl').read_text().splitlines()] for s,p in folders.items()}
    cursor={s:0 for s in sides};pairs={p:[] for p in PHASES};noise={s:{p:[] for p in PHASES} for s in sides};first={}
    for start in range(0,len(corpus),count):
        group=corpus[start:start+count];refs=read(folders['M']/f'references-{start}.json')
        assert len(refs)==len(group)
        for side,p in folders.items():assert read(p/f'references-{start}.json')==refs
        for repeat in range(rounds):
            for mode in ['cold','hot']:
                name=f'g{start}-r{repeat}-{mode}'
                responses={s:read(p/f'{name}-responses.json') for s,p in folders.items()}
                intermediate={s:read(p/f'{name}-scores.json') for s,p in folders.items()}
                assert all(len(responses[s])==len(intermediate[s])==len(group) for s in sides)
                for i,(case,ref) in enumerate(zip(group,refs)):
                    cont=ref['output_ids'];assert len(cont)==tokens
                    actual={s:request_scores(p,indices[s],name,i,case,cont,responses[s][i],tp,s=='D') for s,p in folders.items()}
                    for side in sides:
                        assert json.loads(json.dumps(actual[side]))==intermediate[side][i]
                        row=cases[side][cursor[side]];cursor[side]+=1
                        assert (row['case'],row['round'],row['cache'])==(case['id'],repeat,mode)
                        assert row['cached_tokens']==responses[side][i]['meta_info'].get('cached_tokens')
                        for phase in PHASES:
                            refscore=actual['M'][phase]
                            check_statistics(numerical_stats(list(zip(actual[side][phase],refscore))),row[phase])
                            key=side,case['id'],phase
                            if key in first:noise[side][phase].extend(zip(first[key],actual[side][phase]))
                            else:first[key]=actual[side][phase]
                    for phase in PHASES:pairs[phase].extend(zip(actual.get('D',actual['M'])[phase],actual['M'][phase]))
    expected_pairs=count*rounds*2
    for side,p in folders.items():
        assert cursor[side]==len(cases[side])==expected_pairs
        assert len(indices[side])==expected_pairs
        summary=read(p/'summary.json');assert (summary['case_pairs'],summary['rounds'],summary['decode_tokens'])==(expected_pairs,rounds,tokens)
        for phase in PHASES:check_statistics(numerical_stats(noise[side][phase]),summary['repeat_noise'][phase])
        assert inventory(p)==before[side],'Raw evidence changed during audit'
    scores={p:numerical_stats(v) for p,v in pairs.items()}
    expected_counts={'prefill':sum(len(c['input_ids'])-1 for c in corpus)*rounds*2,'prefill_last':expected_pairs,'decode':expected_pairs*(tokens-1)}
    for phase,n in expected_counts.items():
        assert scores[phase]['count']==n
        if not reference_only:check_statistics(scores[phase],read(folders['D']/'summary.json')['scores'][phase])
    return {'passed':True,'case_pairs':expected_pairs,'scores':scores,'reference_only':reference_only,
            'raw_files_rehashed':{s:len(v) for s,v in before.items()},'original_rank_capture_files_reparsed':expected_pairs*tp*2*len(sides),
            'reference_seal_sha256':digest(folders['M']/'seal.json')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--diagnostic-reference-only',action='store_true');args=parser.parse_args();root=args.root
    if args.diagnostic_reference_only:
        result={'passed':True,'scope':'Existing TP2-per-role M-only diagnostic parser replay. Does not establish D/M accuracy or full4P4D acceptance.',
                'precision':{'full':audit(root,'full',4,1,16,2,'source-host-r80.json','pd2diag','r80',True)},'full4P4D_accepted':False}
        filename='pd2-reference-parser-audit-r94.json'
    else:
        assert read(root/'pd4-precision-perf-complete-r33.json')['natural_outputs_equal']
        result={'passed':True,'scope':'Original full4P4D main/GDN D/M precision only; native timing, physical placement, kernel/transport and actual KV capacity remain separate acceptance gates.',
                'precision':{l:audit(root,l,c,r,t,4,'source-host-r86.json') for l,c,r,t in [('full',30,4,256),('gdn',4,1,16)]}}
        assert sum(v['count'] for x in result['precision'].values() for v in x['scores'].values())==114692
        filename='pd4-original-precision-audit-r94.json'
    result['auditor_sha256']=digest(Path(__file__));out=root/filename;assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
