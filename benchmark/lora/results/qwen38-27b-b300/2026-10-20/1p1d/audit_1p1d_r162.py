"""Independent raw SSE, role-specific resident KV, and adapter load audit."""
import argparse,gzip,hashlib,json,math,re,statistics
from pathlib import Path
def readlines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def quantile(xs,q):
 xs=sorted(xs);p=(len(xs)-1)*q;i=int(p);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(p-i)
def metric(raw,name):
 rows=[]
 for line in raw.splitlines():
  if not line.startswith('sglang:'+name+'{'):continue
  labels,value=line.rsplit(' ',1)
  if 'priority="' in labels and 'priority=""' not in labels:continue
  assert 'tp_rank="0"' in labels,labels
  rows.append(float(value))
 assert len(rows)<=1,(name,rows)
 return rows[0] if rows else None
def inspect(root):
 folder=root/'measure-1p1d-r160'
 assert json.loads((folder/'complete.json').read_text())['passed']
 infos=json.loads((folder/'server-info.json').read_text())
 image='sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
 place=json.loads((folder/'placement.json').read_text())
 apps=[s.split(',') for s in place['gpu_apps'].splitlines() if s.strip()]
 heavy=[v for v in apps if float(v[2].strip().split()[0])>100000]; assert len(heavy)==2
 uuid_index={s.split(',')[1].strip():int(s.split(',')[0]) for s in place['inventory'].splitlines()}
 assert {uuid_index[v[1].strip()] for v in heavy}=={0,1}
 configs={}
 for item,role,gpu in zip(place['containers'],['prefill','decode'],[0,1]):
  assert item['Image']==image
  assert item['HostConfig']['DeviceRequests'][0]['DeviceIDs']==['0','1']; assert '--base-gpu-id' in item['Config']['Cmd']; assert item['Config']['Cmd'][item['Config']['Cmd'].index('--base-gpu-id')+1]==str(gpu)
  assert item['HostConfig']['PidMode']=='host'
  cfg=infos[role].get('server_args',infos[role]);configs[role]=cfg
  assert cfg['tp_size']==1 and cfg['disaggregation_mode']==role
  assert cfg['disaggregation_enable_kv_checksum'] is False
  assert cfg['disaggregation_transfer_backend']=='nixl'
  assert cfg['max_loras_per_batch']==16 and cfg['max_loaded_loras']==32
 rows=readlines(folder/'requests.jsonl');groups=readlines(folder/'groups.jsonl')
 verified=0
 for row in rows:
  x=json.loads((folder/(row['label']+'.json')).read_text())
  req=x['decode_request'];pres=x['prefill_response'];events=x['decode_events']
  assert req['input_ids']==x['prefill_request']['input_ids']
  assert len(req['input_ids'])==row['input_tokens']
  assert len(events[-1]['data']['output_ids'])==row['output_tokens']
  assert events[-1]['data']['output_ids']==row['output_ids']
  assert pres['output_ids']==row['output_ids'][:1]
  assert req['bootstrap_room']==x['prefill_request']['bootstrap_room']
  assert row['e2e_s']>0 and row['ttft_s']>=0 and row['tpot_s']>0
  assert events[-1]['data']['meta_info']['finish_reason']['type']=='length'
  verified+=1
 speed=[]
 for nin in [128,2048]:
  for c in [1,32,64]:
   chosen=[g for g in groups if g['label'].startswith(f'speed-in{nin}-c{c}-r')]
   assert len(chosen)==3
   subset=[r for r in rows if any(r['label'].startswith(g['label']+'-i') for g in chosen)]
   assert len(subset)==3*c
   speed.append({'input_tokens':nin,'output_tokens':128 if nin==128 else 256,'concurrency':c,'rounds':3,'requests':len(subset),'output_tokens_s_mean':statistics.mean(g['output_tokens_s'] for g in chosen),'ttft_p95_s':quantile([r['ttft_s'] for r in subset],.95),'tpot_p95_ms':1000*quantile([r['tpot_s'] for r in subset],.95),'e2e_p95_s':quantile([r['e2e_s'] for r in subset],.95)})
 snapshots=[]
 for line in gzip.open(folder/'raw-metrics.jsonl.gz','rt'):
  x=json.loads(line);raw=x['raw'];x={k:v for k,v in x.items() if k!='raw'}
  x['metrics']={k:metric(raw,k) for k in ['kv_used_tokens','kv_available_tokens','kv_evictable_tokens','num_running_reqs','num_queue_reqs','max_total_num_tokens','kv_cache_memory_usage_gb','mamba_used_tokens','mamba_available_tokens','decode_sum_seq_lens']}
  x['raw_sha256']=hashlib.sha256(raw.encode()).hexdigest();snapshots.append(x)
 capacity=[]
 for group in [g for g in groups if g['label'].startswith('capacity-')]:
  subset=[r for r in rows if r['label'].startswith(group['label']+'-i')]
  assert len(subset)==group['concurrency'] and all(r['cached_tokens_prefill']==0 for r in subset)
  eligible=[s for s in snapshots if s['role']=='decode' and group['started_monotonic']<=s['started_monotonic']<=s['ended_monotonic']<=group['ended_monotonic'] and s['metrics']['kv_used_tokens'] is not None]
  assert eligible
  peak=max(eligible,key=lambda x:x['metrics']['kv_used_tokens'])
  full=[s for s in eligible if s['metrics']['num_running_reqs']==group['concurrency'] and s['metrics']['kv_used_tokens']>=group['input_tokens']*group['concurrency']]
  capacity.append({'input_tokens_each':group['input_tokens'],'concurrency':group['concurrency'],'submitted_prompt_tokens':group['input_tokens']*group['concurrency'],'completed_requests':len(subset),'peak_observed_decode_KV_tokens':peak['metrics']['kv_used_tokens'],'running_at_same_peak':peak['metrics']['num_running_reqs'],'same_peak_sample':peak,'full_prompt_batch_resident_samples':len(full),'scope':'Observed resident lower bound; allocated capacity is reported separately per role; P+D are never summed'})
 controls=readlines(folder/'controls.jsonl')
 loaded=[c for c in controls if c['path']=='/load_lora_adapter' and c['request']['lora_name'].startswith('slot-')]
 assert len(loaded)==32 and all(all(v['success'] for v in c['responses']) for c in loaded)
 assert all(set(loaded[-1]['responses'][i]['loaded_adapters'])=={f'slot-{n:02d}' for n in range(32)} for i in [0,1])
 slots=[]
 for count in [1,4,8,16,32]:
  chosen=[g for g in groups if g['label'].startswith(f'slots-n{count}-r')];assert len(chosen)==2
  subset=[r for r in rows if any(r['label'].startswith(g['label']+'-i') for g in chosen)]
  assert len(subset)==2*count
  slots.append({'registered_adapters':32,'GPU_slots_per_role':16,'requested_adapter_identities':count,'native_requests_completed':len(subset),'output_tokens_s_mean':statistics.mean(g['output_tokens_s'] for g in chosen),'e2e_p95_s':quantile([r['e2e_s'] for r in subset],.95)})
 references=json.loads((folder/'slot-reference-outputs.json').read_text());identity_checks=[]
 for ref in references:
  i=int(ref['label'].rsplit('-',1)[1]);mix=next(r for r in rows if r['label']==f'slots-n32-r0-i{i}')
  same=ref['output_ids']==mix['output_ids'][:len(ref['output_ids'])]
  diffs=[]
  for left,right in zip(ref['output_token_logprobs'],mix['output_token_logprobs']):
   assert left[1]==right[1];diffs.append(abs(left[0]-right[0]))
  assert same and len(diffs)==16
  identity_checks.append({'adapter':ref['adapter'],'compared_natural_tokens':len(diffs),'natural_ids_equal':same,'max_output_logprob_difference_nat':max(diffs)})
 assert sum(g['label']=='slots-base-plus16' for g in groups)==1
 allocations={}
 for role in ['prefill','decode']:
  log=root/f'qwen38-1p1d-{role}-r160-before-merge-r161.log'
  text=log.read_text();assert 'KV checksum mismatch' not in text and 'CUDA out of memory' not in text
  matches=re.findall(r'KV Cache is allocated\. dtype: torch.bfloat16, #tokens: (\d+), K size: ([0-9.]+) GB, V size: ([0-9.]+) GB',text)
  assert len(matches)==1,matches
  count,k,v=matches[0];cfg=configs[role]
  assert int(count)==infos[role]['max_total_num_tokens']
  allocations[role]={'actual_allocated_KV_tokens':int(count),'K_GiB_rounded':float(k),'V_GiB_rounded':float(v),'max_request_context_tokens':cfg['context_length'],'max_running_requests_resolved':cfg['max_running_requests'],'mamba_slots':cfg['max_mamba_cache_size'],'LoRA_GPU_slots':cfg['max_loras_per_batch'],'max_loaded_LoRAs':cfg['max_loaded_loras']}
 merged=root/'measure-1p1d-merged-r161'
 comparisons=[]
 if (merged/'complete.json').exists():
  mr=readlines(merged/'requests.jsonl');mg=readlines(merged/'groups.jsonl');md={r['label']:r for r in mr}
  formal=[r for r in rows if r['label'].startswith('speed-')]
  assert len(formal)==582
  for row in formal:assert row['output_ids']==md[row['label']]['output_ids'],row['label']
  for point in speed:
   prefix=f"speed-in{point['input_tokens']}-c{point['concurrency']}-r"
   selected=[g for g in mg if g['label'].startswith(prefix)];assert len(selected)==3
   rate=statistics.mean(g['output_tokens_s'] for g in selected)
   comparisons.append({**point,'merged_output_tokens_s_mean':rate,'dynamic_fraction_of_merged':point['output_tokens_s_mean']/rate})
 result={'passed':True,'runtime_image':image,'native_SSE_requests_audited':verified,'allocated_capacity_per_role':allocations,'speed':speed,'capacity_residency':capacity,'slots':slots,'adapter_identity_checks':identity_checks,'dynamic_vs_merge':comparisons,'formal_natural_D_M_requests_equal':582 if comparisons else None,'limitations':['BF16 deterministic eager; CUDA graphs disabled; direct SGLang native PD client','32 capacity adapters share real A/B tensors but use distinct alpha scales; not 32 separately trained checkpoints','16 configured GPU slots and 32 registered adapters are verified operating points, not hardware maxima','KV capacity is per role; P and D capacities are not additive','32-adapter mix exceeding the 16-slot GPU pool may queue/reload','Independent teacher-forced numerical acceptance is a separate precision report'],'files_sha256':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['complete.json','server-info.json','placement.json','requests.jsonl','groups.jsonl','controls.jsonl','raw-metrics.jsonl.gz']}}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=inspect(a.root)
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k in ['passed','native_SSE_requests_audited','allocated_capacity_per_role','speed','slots','adapter_identity_checks']}))
