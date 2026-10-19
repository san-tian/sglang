"""Read-only audit of D/M/base native controls; distinguish allocation differences."""
import argparse,collections,gzip,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('topology',choices=['tp2','dpa2','tp4','dpa4']);a=p.parse_args();top=a.topology
R=Path(__file__).resolve().parent;tp=2 if top.endswith('2') else 4;dpa=top.startswith('dpa')
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cfg(p):
 x=read(p);return x.get('server_args',x)
expected_gpus={'tp2':'"device=0,5"','dpa2':'"device=3,4"','tp4':'"device=0,3,4,5"','dpa4':'"device=0,3,4,5"'}[top]
configurations={};rows_by_side={};proofs={};traces_by_side={};summaries={};launches={}
for side in ['d','m','base']:
 suffix='r41' if side=='base' else ('r29' if tp==2 else 'r30');directory=R/f'perf-{top}-{side}-{suffix}'
 complete=read(directory/'complete.json');assert complete['requests']==636 and complete['profile']
 configurations[side]=config=cfg(directory/'server-info.json')
 assert config['tp_size']==tp and config['attn_dp_size']==(tp if dpa else 1)
 assert config['context_length']==4096 and config['mem_fraction_static']==.5
 assert config['max_mamba_cache_size']==256 and config['max_running_requests']==32
 assert bool(config['enable_lora'])==(side=='d')
 # Independently inspect the actual host launch rather than assuming the GPU placement.
 name=f'qwen38-{top}-'+(f'base-r41' if side=='base' else (f'{side}-r28' if tp==2 else (f'm-real_r16-r30' if side=='m' else 'd-r30')))
 launchfile=R/f'launch-{name}.json';cmd=read(launchfile)
 assert cmd[cmd.index('--gpus')+1]==expected_gpus,(launchfile,cmd)
 assert cmd[cmd.index('--tp-size')+1]==str(tp)
 assert ('--enable-lora' in cmd)==(side=='d')
 launches[side]={'file':launchfile.name,'sha256':sha(launchfile),'physical_devices':expected_gpus}
 summary=read(directory/'summary.json');assert summary['request_count']==636 and summary['groups']==48 and len(summary['points'])==16
 summaries[side]=summary
 rows=lines(directory/'requests.jsonl');assert len(rows)==len({r['label'] for r in rows})==636
 records={};hashes={}
 for row in rows:
  label=row['label'];file=directory/(label+'.json');raw=read(file);request=raw['request'];events=raw['events'];final=events[-1]['data'];nin,nout=row['input_tokens'],row['output_tokens']
  assert len(request['input_ids'])==nin and request['stream'] is True
  assert request['sampling_params']=={'temperature':0,'max_new_tokens':nout,'ignore_eos':True}
  assert 'custom_logit_processor' not in request
  assert request.get('lora_path')==('perf-A' if side=='d' else None)
  assert len(final['output_ids'])==nout and final['output_ids']==row['output_ids']
  assert final['meta_info']['finish_reason']=={'type':'length','length':nout}
  assert final['meta_info']['prompt_tokens']==nin and final['meta_info']['completion_tokens']==nout
  assert len(events[0]['data']['output_ids'])==1 and events[0]['elapsed_s']==row['ttft_s']
  assert row['e2e_s']>=events[-1]['elapsed_s']
  assert all(math.isfinite(row[k]) and row[k]>0 for k in ['ttft_s','tpot_s','e2e_s'])
  assert math.isclose(row['tpot_s'],(events[-1]['elapsed_s']-events[0]['elapsed_s'])/(nout-1),rel_tol=1e-9,abs_tol=1e-10)
  records[label]={'input_ids':request['input_ids'],'sampling_params':request['sampling_params'],'output_ids':row['output_ids']};hashes[label]=sha(file)
 groups=lines(directory/'groups.jsonl');assert len(groups)==48
 seen=set()
 for group in groups:
  key=(group['input_tokens'],group['concurrency'],group['round'],group['cache']);assert key not in seen;seen.add(key)
  batch=[r for r in rows if (r['input_tokens'],r['concurrency'],r['round'],r['cache'])==key];assert len(batch)==group['concurrency']
  rate=sum(r['output_tokens'] for r in batch)/group['elapsed_s'];assert math.isclose(rate,group['output_throughput_tokens_s'],rel_tol=1e-10)
 assert seen=={(nin,c,r,cache) for nin in [128,2048] for c in [1,4,16,32] for r in range(3) for cache in ['cold','hot']}
 expected_labels={f'in{nin}-out{nout}-c{c}-r{r}-{cache}-i{i}' for nin,nout in [(128,128),(2048,256)] for c in [1,4,16,32] for r in range(3) for cache in ['cold','hot'] for i in range(c)}
 assert set(records)==expected_labels
 proofs[side]=hashes;rows_by_side[side]=records
 traces=sorted((directory/'profile').rglob('*.trace.json.gz'));assert len(traces)==tp
 summarized=read(directory/'kernel-summary.json');listed={Path(t['file']).name:t for t in summarized['traces']};assert set(listed)=={p.name for p in traces}
 result=[]
 for file in traces:
  assert sha(file)==listed[file.name]['sha256']
  with gzip.open(file,'rt') as f:doc=json.load(f)
  names=collections.Counter(e['name'] for e in doc.get('traceEvents',[]) if e.get('cat')=='kernel' and e.get('ph')=='X' and isinstance(e.get('dur'),(int,float)))
  assert sum(names.values())==listed[file.name]['gpu_kernel_count']>0
  result.append({'trace':file.name,'sha256':sha(file),'kernel_events':sum(names.values()),'FP32_merge_gemm':sum(n for k,n in names.items() if 'sgemm' in k.lower()),'merge_round':sum(n for k,n in names.items() if '_round_merged_weight' in k),'collectives':{k:n for k,n in names.items() if any(s in k.lower() for s in ['nccl','allreduce','all_reduce','cross_device','reduce_scatter','all_gather'])}})
  del doc
 traces_by_side[side]=result
 assert all(t['FP32_merge_gemm']==t['merge_round']==0 for t in result) if side!='d' else True
assert rows_by_side['d']==rows_by_side['m'],'D/M natural outputs differ'
for label,row in rows_by_side['d'].items():
 b=rows_by_side['base'][label];assert row['input_ids']==b['input_ids'] and row['sampling_params']==b['sampling_params']
common=['tp_size','attn_dp_size','context_length','dtype','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','page_size','chunked_prefill_size','max_running_requests','lora_merged_bf16_cache_size_mb']
differences={k:{side:configurations[side].get(k) for side in ['d','m','base']} for k in common if len({str(configurations[side].get(k)) for side in ['d','m','base']})>1}
assert not set(differences)-{'max_total_num_tokens'},differences
assert min(configurations[s]['max_total_num_tokens'] for s in ['d','m','base'])>32*(2048+256)
points=[]
for nin in [128,2048]:
 for c in [1,4,16,32]:
  for cache in ['cold','hot']:
   sidepoints={s:next(x for x in summaries[s]['points'] if (x['input_tokens'],x['concurrency'],x['cache'])==(nin,c,cache)) for s in ['d','m','base']}
   points.append({'input_tokens':nin,'concurrency':c,'cache':cache,'controls':sidepoints,'D_over_M_throughput':sidepoints['d']['throughput_tokens_s_mean']/sidepoints['m']['throughput_tokens_s_mean'],'M_over_base_throughput':sidepoints['m']['throughput_tokens_s_mean']/sidepoints['base']['throughput_tokens_s_mean']})
report={'passed':True,'topology':top,'native_requests_per_side':636,'raw_native_payloads_audited':1908,'natural_D_M_outputs_equal':True,'base_output_ids_equal_M_count':sum(rows_by_side['base'][k]['output_ids']==v['output_ids'] for k,v in rows_by_side['m'].items()),'physical_launches':launches,'configurations':{s:{k:c.get(k) for k in common} for s,c in configurations.items()},'configuration_differences':differences,'max_requested_tokens_per_batch':73728,'speed_points':points,'profile_traces':traces_by_side,'raw_native_sha256':proofs,'scope':'Same GPU placement per topology; natural unforced SSE D/M/base controls. Distinct model weights mean base outputs need not match M. Automatic KV pools differ; all exceed the largest requested batch. Other GPU experiments ran concurrently and controls were taken at separate times; observed ratios are not isolated causal speedups or equal-KV capacity evidence. Actual CUDA trace counts are not wall-time attribution.'}
out=R/f'{top}-threeway-audit-r58.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps({'topology':top,'passed':True,'requests_per_side':636,'raw_native_payloads_audited':1908,'config_differences':differences,'c32_cold':[p for p in points if p['concurrency']==32 and p['cache']=='cold'],'kernel_events':{s:sum(t['kernel_events'] for t in traces) for s,traces in traces_by_side.items()}}),flush=True)
