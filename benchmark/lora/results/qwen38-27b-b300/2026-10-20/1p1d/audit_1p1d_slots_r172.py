"""Independent 32-slot / 64-registration operating-point audit."""
import hashlib,json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 f=R/'measure-1p1d-slots32-r169';assert json.loads((f/'complete.json').read_text())['passed']
 assert json.loads((R/'1p1d-slot-sweep-finished-r169.json').read_text())['primary_dynamic_restored']
 infos=json.loads((f/'server-info.json').read_text());rows=lines(f/'requests.jsonl');groups=lines(f/'groups.jsonl');api=lines(f/'controls.jsonl')
 loads=[x for x in api if x['path']=='/load_lora_adapter'];assert len(loads)==64
 names={f'slot-{i:02d}' for i in range(64)}
 assert all(set(loads[-1]['responses'][role]['loaded_adapters'])==names for role in [0,1])
 assert all(all(r['success'] for r in x['responses']) for x in loads)
 scaled=json.loads((f/'scaled-adapters.json').read_text());assert len({x['alpha'] for x in scaled})==64
 for row in rows:
  raw=json.loads((f/(row['label']+'.json')).read_text());event=raw['decode_events'][-1]['data']
  assert row['output_ids']==event['output_ids'] and len(event['output_ids'])==row['output_tokens']
  assert raw['prefill_response']['output_ids']==row['output_ids'][:1]
  assert event['meta_info']['finish_reason']['type']=='length'
  assert len(raw['decode_request']['input_ids'])==128
 assert len(rows)==262
 points=[]
 for count in [1,16,32,64]:
  gs=[g for g in groups if g['label'].startswith(f'slots-n{count}-r')];assert len(gs)==2
  rr=[r for r in rows if any(r['label'].startswith(g['label']+'-i') for g in gs)];assert len(rr)==2*count
  assert all(g['distinct_adapters']==count for g in gs)
  points.append({'GPU_slots_per_role':32,'registered_adapters_per_role':64,'distinct_requested_adapters':count,'rounds':2,'natural_requests_completed':len(rr),'output_tokens_s_mean':statistics.mean(g['output_tokens_s'] for g in gs),'e2e_s_max':max(x['e2e_s'] for x in rr)})
 refs=json.loads((f/'slot-reference-outputs.json').read_text());checks=[]
 for ref in refs:
  index=int(ref['label'].rsplit('-',1)[1]);other=next(r for r in rows if r['label']==f'slots-n64-r0-i{index}')
  assert ref['output_ids']==other['output_ids'][:16]
  assert all(a[1]==b[1] for a,b in zip(ref['output_token_logprobs'],other['output_token_logprobs']))
  differences=[abs(a[0]-b[0]) for a,b in zip(ref['output_token_logprobs'],other['output_token_logprobs'])]
  assert len(differences)==16
  checks.append({'adapter':ref['adapter'],'natural_ids_equal':True,'scores_compared':16,'max_logprob_difference_nat':max(differences)})
 allocations={}
 for role in ['prefill','decode']:
  cfg=infos[role].get('server_args',infos[role]);assert cfg['max_loras_per_batch']==32 and cfg['max_loaded_loras']==64
  log=(R/f'qwen38-1p1d-slots32-{role}-r169-finished-r169.log').read_text()
  assert 'CUDA out of memory' not in log
  match=re.search(r'KV Cache is allocated\. dtype: torch.bfloat16, #tokens: (\d+), K size: ([0-9.]+) GB, V size: ([0-9.]+) GB',log);assert match
  assert int(match[1])==infos[role]['max_total_num_tokens']
  allocations[role]={'allocated_KV_tokens':int(match[1]),'K_GiB_rounded':float(match[2]),'V_GiB_rounded':float(match[3]),'context_length':cfg['context_length'],'max_running_requests_configured':cfg['max_running_requests'],'mamba_slots':cfg['max_mamba_cache_size']}
 assert any(g['label']=='slots-base-plus32' and g['concurrency']==33 for g in groups)
 report={'passed':True,'natural_requests_audited':262,'GPU_slots_per_role':32,'registered_adapters_per_role':64,'allocated_KV_per_role':allocations,'speed_points':points,'identity_checks':checks,'base_plus32_completed_requests':33,'primary_restored_with_GPU_slots':16,'limitations':['32 GPU slots and 64 registrations are tested operating points, not hardware maxima','Capacity adapters have distinct scales of common real rank16 A/B weights, with a rank32 pool','Over-pool 64-adapter requests can queue and reload','This is 128-input 32-output capacity-mix timing, not the 128/128 or 2048/256 main speed matrix'],'files_sha256':{n:hashlib.sha256((f/n).read_bytes()).hexdigest() for n in ['complete.json','server-info.json','placement.json','requests.jsonl','groups.jsonl','controls.jsonl','scaled-adapters.json']}}
 target=R/'1p1d-slot-sweep-audit-r172.json'
 with target.open('x') as output:json.dump(report,output,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
