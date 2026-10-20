"""Original SSE/percentile/rank-kernel audit for the complete8GPU topology controls."""
import argparse,collections,gzip,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('topology',choices=['tp8','dcp8x2','dpa8','tp4','dpa4']);p.add_argument('--diagnostic-legacy-topology',action='store_true');a=p.parse_args();top=a.topology
legacy=top in ['tp4','dpa4'];assert legacy==a.diagnostic_legacy_topology
R=Path(__file__).resolve().parent;tp=4 if legacy else 8;dpa=top.startswith('dpa');dcp=2 if top=='dcp8x2' else 1
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cfg(p):
 x=read(p);return x.get('server_args',x)
expected_gpus='"device=0,3,4,5"' if legacy else '"device=0,1,2,3,4,5,6,7"'
IMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
placements=[] if legacy else lines(R/'actual-placement-observations-r130.jsonl')
placement_proofs={};latency_statistics=0
configurations={};rows_by_side={};proofs={};traces_by_side={};summaries={};launches={}
for side in ['d','m','base']:
 suffix='r130' if legacy and side=='base' else 'r130';directory=R/f'perf-{top}-{side}-{suffix}'
 complete=read(directory/'complete.json');assert complete['requests']==636 and complete['profile']
 configurations[side]=config=cfg(directory/'server-info.json')
 assert config['tp_size']==tp and config['attn_dp_size']==(tp if dpa else 1)
 assert config['context_length']==4096 and config['mem_fraction_static']==.5
 assert config['max_mamba_cache_size']==(32 if dpa and tp==8 else 256) and config['max_running_requests']==32
 assert config['dcp_size']==dcp
 if dcp>1:assert config['dcp_comm_backend']=='a2a'
 assert bool(config['enable_lora'])==(side=='d')
 # Independently inspect the actual host launch rather than assuming the GPU placement.
 if legacy:name=f'qwen38-{top}-'+('base-r130' if side=='base' else ('m-real_r16-r130' if side=='m' else 'd-r130'))
 else:name=f'qwen38-{top}-'+('base-r130' if side=='base' else ('m-real_r16-'+('r130' if top=='tp8' else 'r130') if side=='m' else 'd-r130'))
 launchfile=R/f'launch-{name}.json';cmd=read(launchfile)
 assert cmd[cmd.index('--gpus')+1]==expected_gpus,(launchfile,cmd)
 assert cmd[cmd.index('--tp-size')+1]==str(tp)
 assert ('--enable-lora' in cmd)==(side=='d')
 launches[side]={'file':launchfile.name,'sha256':sha(launchfile),'physical_devices':expected_gpus}
 if not legacy:
  assert IMAGE in cmd
  matches=[v for v in placements if v['name']==name];assert len(matches)==1
  v=matches[0];assert v['placement_matched'] and v['image']==IMAGE and v['observed_compute_gpus']==v['expected_compute_gpus']==list(range(8))
  assert len({p['pid'] for p in v['scheduler_gpu_processes']})==8
  placement_proofs[side]=v
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
  previous=[];clock=-1;last_increase=None
  for e in events:
   ids=e['data']['output_ids'];t=e['elapsed_s'];assert math.isfinite(t) and t>=clock and ids[:len(previous)]==previous and len(ids)>=len(previous)
   if len(ids)>len(previous):last_increase=t
   previous,clock=ids,t
  assert row['e2e_s']>=events[-1]['elapsed_s']
  assert all(math.isfinite(row[k]) and row[k]>0 for k in ['ttft_s','tpot_s','e2e_s'])
  assert math.isclose(row['tpot_s'],(last_increase-events[0]['elapsed_s'])/(nout-1),rel_tol=1e-9,abs_tol=1e-10)
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
 result=[];rank_ids=[]
 for file in traces:
  assert sha(file)==listed[file.name]['sha256']
  with gzip.open(file,'rt') as f:doc=json.load(f)
  kernels=[e for e in doc.get('traceEvents',[]) if e.get('cat')=='kernel' and e.get('ph')=='X' and isinstance(e.get('dur'),(int,float))]
  assert all(math.isfinite(e['dur']) and e['dur']>=0 for e in kernels)
  names=collections.Counter(e['name'] for e in kernels)
  rank=int(file.name.split('-TP-')[1].split('.')[0].split('-')[0]);rank_ids.append(rank)
  assert sum(names.values())==listed[file.name]['gpu_kernel_count']>0
  result.append({'rank':rank,'all_exact_kernel_counts':dict(names),'trace':file.name,'sha256':sha(file),'kernel_events':sum(names.values()),'FP32_merge_gemm':sum(n for k,n in names.items() if 'sgemm' in k.lower()),'merge_round':sum(n for k,n in names.items() if '_round_merged_weight' in k),'collectives':{k:n for k,n in names.items() if any(s in k.lower() for s in ['nccl','allreduce','all_reduce','cross_device','reduce_scatter','all_gather'])}})
  del doc
 assert len(rank_ids)==tp and set(rank_ids)==set(range(tp))
 traces_by_side[side]=result
 assert all(t['merge_round']==0 for t in result) if side!='d' else True
 for point in summary['points']:
  key=(point['input_tokens'],point['concurrency'],point['cache']);rr=[v for v in rows if (v['input_tokens'],v['concurrency'],v['cache'])==key];gg=[v for v in groups if (v['input_tokens'],v['concurrency'],v['cache'])==key]
  assert len(rr)==point['requests']==3*key[1] and len(gg)==3 and {g['round'] for g in gg}=={0,1,2}
  assert math.isclose(math.fsum(g['output_throughput_tokens_s'] for g in gg)/3,point['throughput_tokens_s_mean'],rel_tol=1e-10)
  for metric in ['ttft_s','tpot_s','e2e_s']:
   values=sorted(r[metric] for r in rr);stats={'mean':math.fsum(values)/len(values),**{f'p{int(q*100)}':values[math.ceil(len(values)*q)-1] for q in [.5,.95,.99]}}
   for k,v in stats.items():assert math.isfinite(point[metric][k]) and math.isclose(v,point[metric][k],rel_tol=1e-10,abs_tol=1e-10);latency_statistics+=1
assert rows_by_side['d']==rows_by_side['m'],'D/M natural outputs differ'
for label,row in rows_by_side['d'].items():
 b=rows_by_side['base'][label];assert row['input_ids']==b['input_ids'] and row['sampling_params']==b['sampling_params']
common=['tp_size','attn_dp_size','context_length','dtype','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','page_size','chunked_prefill_size','max_running_requests','lora_merged_bf16_cache_size_mb','dcp_size','dcp_comm_backend']
differences={k:{side:configurations[side].get(k) for side in ['d','m','base']} for k in common if len({str(configurations[side].get(k)) for side in ['d','m','base']})>1}
assert not set(differences)-{'max_total_num_tokens'},differences
assert min(configurations[s]['max_total_num_tokens'] for s in ['d','m','base'])>32*(2048+256)
points=[]
for nin in [128,2048]:
 for c in [1,4,16,32]:
  for cache in ['cold','hot']:
   sidepoints={s:next(x for x in summaries[s]['points'] if (x['input_tokens'],x['concurrency'],x['cache'])==(nin,c,cache)) for s in ['d','m','base']}
   points.append({'input_tokens':nin,'concurrency':c,'cache':cache,'controls':sidepoints,'D_over_M_throughput':sidepoints['d']['throughput_tokens_s_mean']/sidepoints['m']['throughput_tokens_s_mean'],'M_over_base_throughput':sidepoints['m']['throughput_tokens_s_mean']/sidepoints['base']['throughput_tokens_s_mean']})
assert latency_statistics==576
report={'passed':True,'full8GPU_accepted':not legacy,'latency_statistics_recomputed':latency_statistics,'actual_placement_evidence':placement_proofs,'topology':top,'native_requests_per_side':636,'raw_native_payloads_audited':1908,'natural_D_M_outputs_equal':True,'base_output_ids_equal_M_count':sum(rows_by_side['base'][k]['output_ids']==v['output_ids'] for k,v in rows_by_side['m'].items()),'physical_launches':launches,'configurations':{s:{k:c.get(k) for k in common} for s,c in configurations.items()},'configuration_differences':differences,'max_requested_tokens_per_batch':73728,'speed_points':points,'profile_traces':traces_by_side,'raw_native_sha256':proofs,'scope':'Candidate R116 image, new R118 namespace; original R100 raw numerical reconstruction retained. Three natural636-request controls,48speed points/576latency stats and3*TP actual rank traces fully reparsed and hashed. New8GPU placement checked via original cgroup/scheduler/GPU UUID observations; legacy4GPU replay validates parsers only. AutomaticKV pools differ, all exceed requested73728tokens; not equal-byte residency evidence or whole-GPU maxcapacity. Controls at separate times; ratios are observations, not isolated causal speedups. Kernel counts/names alone are not E2E time attribution; FP32sgemm name is reported without assuming every such event is a weight merge.'}
out=R/f'{top}-native-kernel-raw-audit-r130.json';assert not out.exists();out.write_text(json.dumps(report,indent=2))
print(json.dumps({'topology':top,'passed':True,'requests_per_side':636,'raw_native_payloads_audited':1908,'config_differences':differences,'c32_cold':[p for p in points if p['concurrency']==32 and p['cache']=='cold'],'kernel_events':{s:sum(t['kernel_events'] for t in traces) for s,traces in traces_by_side.items()}}),flush=True)
