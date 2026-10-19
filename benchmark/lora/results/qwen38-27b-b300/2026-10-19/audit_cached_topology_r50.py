"""Rehash independent merge references and audit every case before reporting acceptance."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('topology',choices=['cache-tp4-r50','cache-dpa4-r50']);a=p.parse_args()
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');top=a.topology
out={'topology':top,'accuracy':{},'performance':{'accepted':False,'reason':'Both complete performance sides and raw requests are required'},'scope':'Same-topology independent merge. Prefix-cache restoration, maximum resident KV and maximum context require separate experiments.'}
for label,pairs,nscore in [('full',240,111184),('gdn',8,3508)]:
 m=R/f'{top}-{label}-reference-r30';d=R/f'{top}-{label}-dynamic-r30'
 seal=json.loads((m/'seal.json').read_text())
 for filename,sha in seal['files'].items():assert hashlib.sha256((m/filename).read_bytes()).hexdigest()==sha,(m,filename)
 ds=json.loads((d/'summary.json').read_text());rs=[json.loads(line) for line in (d/'cases.jsonl').read_text().splitlines()]
 assert len(rs)==ds['case_pairs']==pairs and sum(x['count'] for x in ds['scores'].values())==nscore
 for phase,s in ds['scores'].items():
  assert s['passes_logprob_gate'] and s['mean_abs_nat']<=.01 and s['p99_abs_nat']<=.1,(phase,s)
  assert all(row[phase]['passes_logprob_gate'] and row[phase]['mean_abs_nat']<=.01 and row[phase]['p99_abs_nat']<=.1 for row in rs)
 for directory in [m,d]:
  noise=json.loads((directory/'summary.json').read_text())['repeat_noise']
  assert all(v['passes_logprob_gate'] for v in noise.values()),(directory,noise)
 out['accuracy'][label]={'case_pairs':pairs,'score_count':nscore,'scores':ds['scores'],'sealed_reference_files_rehashed':len(seal['files']),'reference_seal_sha256':hashlib.sha256((m/'seal.json').read_bytes()).hexdigest(),'per_case_gate_verified':True,'dynamic_repeat_noise':ds['repeat_noise'],'reference_repeat_noise':json.loads((m/'summary.json').read_text())['repeat_noise']}
perfpaths={side:R/(f'perf-pd4-{side}-r33' if top=='pd4-r33' else f'perf-{top}-{side}-r30') for side in ['m','d']}
if all((q/'summary.json').exists() for q in perfpaths.values()):
 rows={side:[json.loads(x) for x in (directory/'requests.jsonl').read_text().splitlines()] for side,directory in perfpaths.items()}
 summaries={side:json.loads((q/'summary.json').read_text()) for side,q in perfpaths.items()}
 assert all(len(v)==636 and len({row['label'] for row in v})==636 for v in rows.values())
 assert all(s['request_count']==636 and s['groups']==48 and len(s['points'])==16 for s in summaries.values())
 ref={row['label']:row['output_ids'] for row in rows['m']};wrong=[row['label'] for row in rows['d'] if row['output_ids']!=ref[row['label']]]
 assert not wrong,wrong
 out['performance']={'accepted':True,'requests_per_side':636,'natural_output_mismatches':wrong,'summaries':summaries,'scope':'Absolute throughput/latency under recorded concurrent background experiments; not an isolated causal comparison against another topology'}
(R/f'{top}-independent-audit-r40.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'topology':top,'accuracy_pairs':sum(x['case_pairs'] for x in out['accuracy'].values()),'score_count':sum(x['score_count'] for x in out['accuracy'].values()),'reference_files_rehashed':sum(x['sealed_reference_files_rehashed'] for x in out['accuracy'].values()),'maximum_abs_nat':max(s['max_abs_nat'] for x in out['accuracy'].values() for s in x['scores'].values()),'performance_accepted':out['performance']['accepted']}))

# Current runtime on/off/M/base controls, with actual completed traces.
keys=['tp_size','dp_size','attn_dp_size','pp_size','dtype','kv_cache_dtype','context_length','max_req_input_len','chunked_prefill_size','max_running_requests','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','linear_attn_backend','linear_attn_prefill_backend','linear_attn_decode_backend','bf16_gemm_backend','enable_deterministic_inference','disable_cuda_graph','mamba_ssm_dtype','mamba_radix_cache_strategy','page_size']
folders={side:R/f'perf-{top}-{side}-r30' for side in ['m','d','nocache','base']}
configs={side:json.loads((folder/'server-info.json').read_text()) for side,folder in folders.items()}
reference={row['label']:row for row in [json.loads(line) for line in (folders['m']/'requests.jsonl').read_text().splitlines()]}
inputs={label:json.loads((folders['m']/(label+'.json')).read_text())['request']['input_ids'] for label in reference}
control_summaries={};profile_counts={};hashes={};times=[]
for side,folder in folders.items():
 config=configs[side];assert config['max_total_num_tokens']==65536 and config['max_mamba_cache_size']==256
 assert not {key:(config.get(key),configs['m'].get(key)) for key in keys if config.get(key)!=configs['m'].get(key)}
 assert config['lora_merged_bf16_cache_size_mb']==((12288 if 'dpa' in top else 10240) if side=='d' else 0)
 summary=json.loads((folder/'summary.json').read_text());assert summary['request_count']==636 and len(summary['points'])==16 and summary['groups']==48
 assert (folder/'complete.json').exists()
 rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()];assert len(rows)==len({row['label'] for row in rows})==636
 for row in rows:
  native=folder/(row['label']+'.json');raw=json.loads(native.read_text())
  assert raw['request']['input_ids']==inputs[row['label']]
  assert raw['events'][-1]['data']['output_ids']==row['output_ids']
  assert len(row['output_ids'])==row['output_tokens']
  if side!='base':assert row['output_ids']==reference[row['label']]['output_ids'],(side,row['label'])
  end=native.stat().st_mtime;times.append((end-row['e2e_s'],end))
 traces=json.loads((folder/'kernel-summary.json').read_text())['traces'];assert len(traces)==4
 profile_counts[side]=[]
 for trace in traces:
  path=folder/'profile'/Path(trace['file']).name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==trace['sha256']
  categories={row['name']:row['count'] for row in trace['categories']}
  if side in ['d','m','base']:assert categories.get('FP32 GEMM',0)==categories.get('LoRA scale/add/BF16 store',0)==0,(side,categories)
  profile_counts[side].append({'trace':path.name,'sha256':trace['sha256'],'kernel_events':trace['gpu_kernel_count'],'categories':categories})
 for file in ['summary.json','requests.jsonl','groups.jsonl','server-info.json','complete.json','kernel-summary.json']:
  hashes[str((folder/file).relative_to(R))]=hashlib.sha256((folder/file).read_bytes()).hexdigest()
 control_summaries[side]=summary
observations=json.loads((R/f'{top}-gpu-observation-snapshot.json').read_text());assert observations
assert not any(sample['unexpected'] for sample in observations)
assert observations[0]['unix_s']<=min(start for start,end in times) and observations[-1]['unix_s']>=max(end for start,end in times)-2
assert all(len(sample['gpus'])==8 and all(gpu['memory_mib']<2000 and gpu['utilization_pct']==0 for gpu in sample['gpus'] if gpu['index'] not in [0,1,2,3]) for sample in observations)
result={'topology':top,'scope':'Same physical GPUs0..3/R34/KV65536/SSM256; cached and uncached natural outputs636 each match independent M; base636. Cache main240/GDN8 independent-merge accuracy audited separately. Other4GPUs idle in every1s sample. No concurrent topology workloads; no claim of max KV residency or maximum context from these tests. UncachedD has natural equivalence, not a separate forced-score primary matrix.','controls':control_summaries,'per_rank_profiles':profile_counts,'source_files_sha256':hashes,'matching_resolved_config':{key:configs['m'].get(key) for key in keys},'gpu_observation_samples':len(observations),'accuracy_audit':f'{top}-independent-audit-r40.json'}
(R/f'{top}-fourway-comparison-r50.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'four_controls636requests_each':True,'all_natural_dynamic_outputs_match_M':True,'rank_traces':{side:len(value) for side,value in profile_counts.items()},'gpu_samples':len(observations)}))
