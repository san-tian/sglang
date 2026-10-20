"""Independently recompute FP8 figures from raw responses and same-scrape metrics."""
import gzip,hashlib,json,math,re,statistics,time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(raw,name):
 matches=[]
 for line in raw.splitlines():
  if not line.startswith('sglang:'+name+'{'):continue
  left,val=line.rsplit(' ',1);lab=dict(re.findall(r'(\w+)="([^\"]*)"',left))
  if lab.get('priority','')!='':continue
  matches.append((int(lab.get('tp_rank','0')),float(val)))
 return min(matches)[1] if matches else None

def audit(kind):
 f=E/kind;result={'variant':kind,'phases':{},'request_raw_integrity_passed':True};allrequests=[]
 for phase in ['smoke','speed','capacity','slots','natural','residency-paced']:
  folder=f/phase
  if not folder.exists():continue
  reqs=rows(folder/'requests.jsonl');groups=rows(folder/'groups.jsonl');allrequests.extend(reqs)
  for row in reqs:
   raw=json.loads((folder/(row['label']+'.json')).read_text());last=raw['events'][-1]['data'];q=raw['request']
   assert len(q['input_ids'])==row['input_tokens'];assert last['output_ids']==row['output_ids'] and len(last['output_ids'])==row['output_tokens'];assert last['meta_info']['finish_reason']['type']=='length'
   if kind=='pd':assert raw['prefill_response']['output_ids']==last['output_ids'][:1] and raw['prefill_request']['bootstrap_room']==q['bootstrap_room']
  for g in groups:
   rs=[r for r in reqs if r['label'].startswith(g['label']+'-i')];assert len(rs)==g['concurrency'];assert all(r['input_tokens']==g['input_tokens'] and r['output_tokens']==g['output_tokens'] for r in rs)
   assert math.isclose(g['output_tokens_s'],sum(r['output_tokens'] for r in rs)/g['elapsed_s'],rel_tol=1e-12)
  info=json.loads((folder/'server-info.json').read_text()) if (folder/'server-info.json').exists() else {}
  rp={'complete':(folder/'complete.json').exists(),'failure':json.loads((folder/'failed.json').read_text()) if (folder/'failed.json').exists() else None,'verified_completed_requests':len(reqs),'reported_KV_tokens_per_role':{r:x['max_total_num_tokens'] for r,x in info.items()}}
  if phase=='speed':
   rp['speed']=[]
   for nin,c in [(128,1),(128,32),(128,64),(2048,32)]:
    gs=[g for g in groups if g['label'].startswith(f'speed-in{nin}-c{c}-r')];rs=[r for r in reqs if r['label'].startswith(f'speed-in{nin}-c{c}-r')]
    if not gs:continue
    a=[g['output_tokens_s'] for g in gs];meta=[r['meta_info'] for r in rs];nverify=sum(m.get('spec_verify_ct',0) for m in meta);proposed=sum(m.get('spec_num_proposed_drafts',0) for m in meta);correct=sum(m.get('spec_num_correct_drafts',0) for m in meta)
    rp['speed'].append({'input_tokens':nin,'output_tokens':gs[0]['output_tokens'],'concurrency':c,'rounds':len(gs),'output_tokens_s_rounds':a,'output_tokens_s_mean':statistics.mean(a),'TTFT_mean_s':statistics.mean(r['ttft_s'] for r in rs),'TPOT_mean_s':statistics.mean(r['tpot_s'] for r in rs),'spec_verify_ct':nverify,'spec_accept_rate':correct/proposed if proposed else None,'spec_accept_length':sum(r['output_tokens'] for r in rs)/nverify if nverify else None})
  if phase in ['capacity','residency-paced']:
   ms=[]
   for line in gzip.open(folder/'raw-metrics.jsonl.gz','rt'):
    s=json.loads(line)
    if s['role'] not in ['decode','colocated']:continue
    raw=s.pop('raw');s['raw_sha256']=hashlib.sha256(raw.encode()).hexdigest();s['metrics']={k:metric(raw,k) for k in ['kv_used_tokens','num_running_reqs','decode_sum_seq_lens','max_total_num_tokens']};ms.append(s)
   rp['capacity']=[]
   for g in groups:
    samples=[s for s in ms if g['started_monotonic']<=s['started_monotonic']<=s['ended_monotonic']<=g['ended_monotonic'] and s['metrics']['decode_sum_seq_lens'] is not None and (s['metrics']['num_running_reqs'] or 0)>0]
    full=[s for s in samples if s['metrics']['num_running_reqs']==g['concurrency'] and s['metrics']['decode_sum_seq_lens']>=g['concurrency']*g['input_tokens']]
    rp['capacity'].append({**g,'total_context_tokens_each':g['input_tokens']+g['output_tokens'],'eligible_samples':len(samples),'full_batch_resident_samples':len(full),'full_batch_residency_gate_passed':len(full)>=2,'max_running_requests_observed':max(s['metrics']['num_running_reqs'] for s in samples) if samples else None,'same_scrape_peak_active_sequences':max(samples,key=lambda s:s['metrics']['decode_sum_seq_lens']) if samples else None})
  if phase=='slots':
   for n in ['registration-33.json','slot-output-comparison.json']:
    if (folder/n).exists():rp[n]=json.loads((folder/n).read_text())
   if (folder/'loaded-32-models.json').exists():rp['model_ids_at_32']={r:[z['id'] for z in m['data']] for r,m in json.loads((folder/'loaded-32-models.json').read_text()).items()}
  result['phases'][phase]=rp
 result['verified_total_requests']=len(allrequests);result['files_sha256']={str(p.relative_to(f)):sha(p) for p in f.rglob('*') if p.is_file()}
 return result
if __name__=='__main__':
 report={'unix_s':time.time(),'scope':'Single physical GPU2 requested exact flags; GPUs4,5 PD changes lazy strategy to extra_buffer; GPU3 MTP-off control. Raw native greedy synthetic throughput and token pressure; no quantized-merge precision or long-task quality certification.','variants':[audit(k) for k in ['colocated','pd','off']]}
 by={x['variant']:x for x in report['variants']};on=by['colocated']['phases'].get('speed',{}).get('speed',[]);off=by['off']['phases'].get('speed',{}).get('speed',[])
 report['colocated_MTP_speedup_vs_off']=[{'input_tokens':a['input_tokens'],'concurrency':a['concurrency'],'ratio':a['output_tokens_s_mean']/b['output_tokens_s_mean']} for a,b in zip(on,off)]
 for role in ['colocated','pd']:
  d=E/role/'speed';f=E/'off'/'speed';ref={r['label']:r for r in rows(f/'requests.jsonl')};rs=[r for r in rows(d/'requests.jsonl') if r['label'].startswith('speed-') and r['label'] in ref]
  for r in rs:
   a=json.loads((d/(r['label']+'.json')).read_text())['request'];b=json.loads((f/(r['label']+'.json')).read_text())['request'];assert a['input_ids']==b['input_ids']
  report[role+'_formal_natural_vs_MTP_off']={'requests':len(rs),'whole_outputs_equal':sum(r['output_ids']==ref[r['label']]['output_ids'] for r in rs),'note':'Same synthetic speed prompts; not model-quality or LoRA-vs-merge acceptance.'}
 # Short natural continuation and prefill-score comparisons; no full forced decode claim.
 def stats(pairs):
  if not pairs:return {'count':0}
  ds=sorted(abs(float(a[0])-float(b[0])) for a,b in pairs);assert all(math.isfinite(d) for d in ds)
  return {'count':len(ds),'mean_abs_nat':statistics.mean(ds),'p99_abs_nat':ds[math.ceil(.99*len(ds))-1],'max_abs_nat':max(ds),'argmax_matches':sum(a[1]==b[1] for a,b in pairs),'passes_old_gate':statistics.mean(ds)<=.01 and ds[math.ceil(.99*len(ds))-1]<=.1}
 def scores(row,field):
  m=row.get('prefill_meta_info',row['meta_info']) if field=='input_top_logprobs' else row['meta_info']
  if field=='input_top_logprobs':return [(v[0],t[0][1]) for v,t in zip(m['input_token_logprobs'][1:],m['input_top_logprobs'][1:])]
  return [(v[0],t[0][1]) for v,t in zip(m['output_token_logprobs'],m['output_top_logprobs'])]
 nats={k:{r['label']:r for r in rows(E/k/'natural'/'requests.jsonl')} for k in ['colocated','pd','off']}
 for row in nats['pd'].values():
  raw=json.loads((E/'pd/natural'/(row['label']+'.json')).read_text());row['prefill_meta_info']=raw['prefill_response']['meta_info']
 report['short_natural_comparisons']={}
 for got,want in [('colocated','off'),('pd','off'),('pd','colocated')]:
  pairs={k:[] for k in ['prefill','first_token','decode_common_prefix']};whole=0;cases=[]
  for label,a in nats[got].items():
   if label not in nats[want]:continue
   b=nats[want][label];ap,bp=scores(a,'input_top_logprobs'),scores(b,'input_top_logprobs');assert len(ap)==len(bp);pairs['prefill'].extend(zip(ap,bp))
   ao,bo=scores(a,'output_top_logprobs'),scores(b,'output_top_logprobs');assert len(ao)==len(bo)==32;pairs['first_token'].append((ao[0],bo[0]))
   lcp=next((i for i,(x,y) in enumerate(zip(a['output_ids'],b['output_ids'])) if x!=y),32)
   pairs['decode_common_prefix'].extend(zip(ao[1:lcp],bo[1:lcp]));whole+=lcp==32;cases.append({'label':label,'equal_prefix_tokens':lcp})
  report['short_natural_comparisons'][got+'_vs_'+want]={'requests':len(cases),'whole_outputs_equal':whole,'scores':{k:stats(v) for k,v in pairs.items()},'prefixes':cases,'scope':'8 short prompts x 2 rounds x cold/hot; unforced natural outputs; decode scores restricted to identical prior token prefixes, excluding first divergence so the scored token is identical. Not full teacher-forced LoRA-vs-merge acceptance.'}
 ref=R/'precision-1p1d-reference-r168'
 seal=json.loads((ref/'seal.json').read_text());assert all(sha(ref/n)==v for n,v in seal['files'].items())
 report['BF16_reference_seal_sha256']=sha(ref/'seal.json')
 for kind in ['colocated','pd','off']:
  ps=[];refs=json.loads((ref/'references-0.json').read_text());whole=0;count=0
  for label,a in nats[kind].items():
   parts=label.split('-');repeat=int(parts[1][1:]);cache=parts[2];i=int(parts[3][1:]);expected=json.loads((ref/f'g0-r{repeat}-{cache}-scores.json').read_text())[i]['prefill'];actual=scores(a,'input_top_logprobs');assert len(actual)==len(expected);ps.extend(zip(actual,expected));whole+=a['output_ids']==refs[i]['output_ids'];count+=1
  report['short_natural_comparisons'][kind+'_vs_frozen_BF16_independent_merge']={'requests':count,'whole_outputs_equal':whole,'prefill_scores':stats(ps),'scope':'Combined weight/activation/KV/SSM/kernel/LoRA-backend change relative to old BF16 independent merge. Does not isolate LoRA error.'}
 (E/'audit-r247.json').write_text(json.dumps(report,indent=2));print(json.dumps({'variants':[{k:v[k] for k in ['variant','verified_total_requests']} for v in report['variants']],'MTP_speedups':report['colocated_MTP_speedup_vs_off']}))
