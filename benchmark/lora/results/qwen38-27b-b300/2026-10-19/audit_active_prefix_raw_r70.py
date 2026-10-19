"""Independently reparse R55 real prefix hits and original pre-forcing scores."""
import json,math
from pathlib import Path
from audit_active_raw_r65 import read,digest,inventory,numerical_stats,check_statistics
R=Path(__file__).resolve().parent
output=R/'active-prefix-raw-audit-r70.json';assert not output.exists()
D=R/'cache-prefix-dynamic-r56';M=R/'cache-prefix-reference-r56'
paths=[D]+[M/v for v in ('real','base','gdn')]
snapshots={str(p.relative_to(R)):inventory(p) for p in paths}
for p in paths[1:]:
 seal=read(p/'seal.json')
 assert all(digest(p/name)==sha for name,sha in seal['files'].items())
cfgs=[]
for p in paths:
 cfg=read(p/'server-info.json');cfg=cfg.get('server_args',cfg)
 assert cfg['tp_size']==cfg['attn_dp_size']==1 and cfg['mem_fraction_static']==.52
 assert cfg['max_total_num_tokens']==65536 and cfg['max_mamba_cache_size']==256
 assert bool(cfg['enable_lora'])==(p==D)
 cfgs.append(cfg)
assert cfgs[0]['lora_merged_bf16_cache_policy']=='active' and cfgs[0]['lora_merged_bf16_cache_size_mb']==36864
summary=read(D/'summary.json');assert summary['prompt_tokens']==1630 and summary['case_pairs']==len(summary['rows'])==10
assert {(r['variant'],r['cache']) for r in summary['rows']}=={(v,c) for v in ('real','base','zero','gdn','twin') for c in ('cold','hot')}

def native(p,key,cache,cont):
 payload=read(p/f'{key}-{cache}-payload.json');response=read(p/f'{key}-{cache}-response.json')
 assert len(payload['input_ids'])==1630 and response['output_ids']==cont and len(cont)==16
 assert payload['sampling_params']['custom_params']['continuation']==cont
 assert payload['sampling_params']['max_new_tokens']==16 and payload['sampling_params']['ignore_eos']
 assert payload['sampling_params']['temperature']==0
 expected_alias='prefix39_'+key if p==D and key!='base' else None
 assert payload.get('lora_path')==expected_alias
 meta=response['meta_info'];assert meta['prompt_tokens']==1630 and meta['completion_tokens']==16
 assert meta['finish_reason']['type']=='length' and meta['cached_tokens']==(1600 if cache=='hot' else 0)
 files=list(p.glob(f'{key}-{cache}-capture.rank-*'));assert len(files)==1 and files[0].name.endswith('.rank-0')
 rows=[json.loads(line) for line in files[0].read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(16)) and [r['token'] for r in rows]==cont
 scores=[(r['logprob'],r['argmax']) for r in rows];assert all(math.isfinite(v) and type(i) is int for v,i in scores)
 assert [list(v) for v in scores]==read(p/f'{key}-{cache}-scores.json')
 return scores,payload['input_ids']

pairs=[];first={};proof=[]
for row in summary['rows']:
 key,cache=row['variant'],row['cache'];variant={'real':'real','twin':'real','base':'base','zero':'base','gdn':'gdn'}[key]
 ref=M/variant;cont=read(ref/'natural.json')['output_ids']
 actual,ids=native(D,key,cache,cont);expected,refids=native(ref,variant,cache,cont);assert ids==refids
 values=list(zip(actual,expected));stats=numerical_stats(values);check_statistics(stats,row['scores'])
 assert stats['max_abs_nat']==0 and stats['argmax_match_fraction']==1
 pairs.extend(values)
 if cache=='cold':first[key]=actual
 else:
  repeat=numerical_stats(list(zip(actual,first[key])));check_statistics(repeat,row['cold_hot_noise'])
  assert repeat['max_abs_nat']==0 and repeat['argmax_match_fraction']==1
 proof.append({'variant':key,'cache':cache,'cached_tokens':1600 if cache=='hot' else 0,'max_abs_nat':stats['max_abs_nat']})
aggregate=numerical_stats(pairs);check_statistics(aggregate,summary['scores']);assert aggregate['count']==160
assert all(inventory(p)==snapshots[str(p.relative_to(R))] for p in paths)
result={'passed':True,'case_pairs':10,'score_count':160,'scores':aggregate,'identity_cache_proofs':proof,
        'raw_reference_and_dynamic_files_rehashed':sum(map(len,snapshots.values())),
        'reference_seal_sha256':{v:digest(M/v/'seal.json') for v in ('real','base','gdn')},
        'dynamic_summary_sha256':digest(D/'summary.json'),'audit_source_sha256':digest(Path(__file__)),
        'score_auditor_dependency_sha256':digest(Path(__file__).with_name('audit_active_raw_r65.py')),
        'scope':'R55 TP1 active cache, mem.52/KV65536/SSM256. Five identities, actual cold0/hot1600. Original last-prompt plus15decode scores, paired to independently merged sealed references (base for zero; same real merge for twin). Raw requests/responses/captures checked on both sides; not full-prefill, lifecycle or speed acceptance.'}
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
