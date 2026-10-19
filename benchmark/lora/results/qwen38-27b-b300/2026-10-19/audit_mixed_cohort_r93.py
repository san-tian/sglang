"""Independently reconstruct one completed R84 natural mixed cohort's timings.

A single cohort is not a cache speedup or full six-run acceptance. Hash every
original request/events file; never include its token IDs or generated content
in the portable report.
"""
import argparse,collections,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=int,required=True);p.add_argument('--policy',choices=['nocache','lru','active'],required=True);a=p.parse_args()
root=a.root;folder=root/f'perf-active-mix-{a.run}-{a.policy}-r84'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(x,y):assert math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8),(x,y)
def quantile(values,q):return sorted(values)[max(0,math.ceil(len(values)*q)-1)]
summary=read(folder/'summary.json');assert summary['request_count']==192 and summary['groups']==6
rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
groups=[json.loads(s) for s in (folder/'groups.jsonl').read_text().splitlines()]
assert len(rows)==len({r['label'] for r in rows})==192 and len(groups)==6
cfg=read(folder/'server-info.json');cfg=cfg.get('server_args',cfg)
assert (cfg['tp_size'],cfg['max_total_num_tokens'],cfg['max_mamba_cache_size'],cfg['mem_fraction_static'])==(1,65536,256,.52)
actual=read(root/f'qwen38-active-mix-{a.run}-{a.policy}-r84-actual-config.json')
assert actual['image']=='sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
assert 'TMPDIR=/root/.cache/tmp' in actual['env']
hashes={};reconstructed={}
for row in rows:
 f=folder/(row['label']+'.json');before=sha(f);raw=read(f);req=raw['request'];events=raw['events']
 assert req['stream'] is True and 'custom_logit_processor' not in req
 assert len(req['input_ids'])==2048 and req['sampling_params']=={'temperature':0,'max_new_tokens':256,'ignore_eos':True}
 assert req.get('lora_path')=={'base':None,'real':'mix-real','half':'mix-half','gdn':'mix-gdn'}[row['identity']]
 prev=0;first=last=None;previous_ids=[];first_count=0;clock=-1
 for ev in events:
  t=ev['elapsed_s'];data=ev['data'];ids=data['output_ids'];n=len(ids)
  assert math.isfinite(t) and t>=clock and ids[:prev]==previous_ids
  if n>prev:
   if first is None:first,first_count=t,n
   last=t
  previous_ids=ids;prev=n;clock=t
 assert prev==256 and first_count==1 and previous_ids==row['output_ids']
 assert data['meta_info']['finish_reason']=={'type':'length','length':256}
 assert data['meta_info']['prompt_tokens']==2048 and data['meta_info']['completion_tokens']==256
 tpot=(last-first)/(256-first_count);close(first,row['ttft_s']);close(tpot,row['tpot_s'])
 assert row['e2e_s']>=last and math.isfinite(row['e2e_s'])
 hashes[f.name]=before;assert sha(f)==before
 reconstructed[row['label']]={'ttft_s':first,'tpot_s':tpot,'e2e_s':row['e2e_s']}
for g in groups:
 subset=[r for r in rows if (r['round'],r['cache'])==(g['round'],g['cache'])]
 assert len(subset)==32 and collections.Counter(r['identity'] for r in subset)=={'base':8,'real':8,'half':8,'gdn':8}
 assert g['elapsed_s']>=max(r['e2e_s'] for r in subset)
 close(g['output_throughput_tok_s'],8192/g['elapsed_s'])
points=[]
for cache in ['cold','hot']:
 subset=[r for r in rows if r['cache']==cache];gs=[g for g in groups if g['cache']==cache]
 assert len(subset)==96 and len(gs)==3
 saved=next(x for x in summary['points'] if x['cache']==cache)
 mean=sum(8192/g['elapsed_s'] for g in gs)/3;close(mean,saved['output_throughput_tok_s'])
 point={'cache':cache,'requests':96,'throughput_tokens_s_mean':mean}
 for metric in ['ttft_s','tpot_s','e2e_s']:
  vals=[reconstructed[r['label']][metric] for r in subset]
  point[metric]={k:quantile(vals,q) for k,q in [('p50',.5),('p95',.95),('p99',.99)]}
  for k,v in point[metric].items():close(v,saved[metric][k])
 hits=[r['cached_tokens'] for r in subset if r['cached_tokens'] is not None]
 point['cached_tokens_min_max']=[min(hits),max(hits)] if hits else None
 points.append(point)
result={'passed':True,'run':a.run,'policy':a.policy,'natural_requests':192,'groups':6,'warmup_batches':len(read(folder/'warmup.json')),'points':points,'original_request_files_sha256':hashes,'summary_sha256':sha(folder/'summary.json'),'actual_runtime_image':actual['image'],'scope':'One completed R84 cohort. Three cold/hot rounds,32 concurrent2048-in/256-out mixed requests,KV65536/SSM256. Client E2E includes connection and stream completion; raw events independently establish TTFT/TPOT. No active/LRU benefit, cross-policy equivalence or six-run acceptance until all cohorts complete.'}
assert result['warmup_batches']==2
(root/f'mixed-cohort-audit-{a.run}-{a.policy}-r93.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='original_request_files_sha256'}))
