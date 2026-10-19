"""Experiment-only scheduling helpers; require verified lease completion and idle GPUs."""
import hashlib,json,shutil,subprocess,time,urllib.request
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
W='/work/qwen38-mol-multigpu-20261019'
CLIENT=W+'/client-r33'
BASE='/work/models/Qwen3.8-27B'
MERGES='/work/qwen38-mol-development-20261019/merged/'
ADAPTERS='/work/qwen38-mol-experiment/adapters/'
def run(cmd,log=None):
 if log:
  with (R/log).open('x') as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return subprocess.run(cmd,check=True)
def waitleases(markers,gpus,timeout=18000):
 end=time.monotonic()+timeout
 while True:
  exists=all((R/n).exists() for n in markers)
  if exists:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
   used={int(a.strip()):int(b.strip()) for a,b in [s.split(',') for s in rows]}
   if all(used[g]<2000 for g in gpus):return
  if time.monotonic()>end:raise TimeoutError({'markers':markers,'gpus':gpus})
  time.sleep(5)
def health(port,name):
 end=time.monotonic()+1800
 while True:
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=5) as f:
    if f.status==200:return
  except Exception:pass
  state=json.loads(subprocess.check_output(['docker','inspect',name]))[0]['State']
  if not state['Running']:raise RuntimeError({'container':name,'state':state})
  if time.monotonic()>end:raise TimeoutError(name)
  time.sleep(2)
def launch(name,gpus,tp,port,model,*,dpa=False,extra=(),memory=.5,mamba=256,requests=32,env=None):
 cache=R/'caches'/name
 assert not cache.exists(),cache
 shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache)
 for rank in range(tp):
  tune=cache/f'sglang/flashinfer/autotune/0.7.2/sm103/add08325a8bdc6e8/rank_tp{rank}_pp0_dp0.json'
  tune.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile('/opt/dlami/nvme/qwen38-mol-fork-20261019/autotune-valid-trace-v2.json',tune)
 cmd=json.loads((R/'launch-qwen38-tp2-d-r28.json').read_text())
 cmd[cmd.index('qwen38-mol:r28')]='sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7'
 if cmd[0]=='sudo':cmd=cmd[1:]
 cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--gpus')+1]='"device='+','.join(map(str,gpus))+'"'
 for flag,value in [('--model-path',model),('--port',port),('--tp-size',tp),('--max-mamba-cache-size',mamba),('--max-running-requests',requests),('--mem-fraction-static',memory)]:cmd[cmd.index(flag)+1]=str(value)
 for i,value in enumerate(cmd):
  if value.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
  if value.startswith('PYTHONPATH='):cmd[i]='PYTHONPATH='+CLIENT
 if model!=BASE:cmd=cmd[:cmd.index('--enable-lora')]
 if dpa:cmd+=['--attn-dp-size',str(tp)]
 for key,value in (env or {}).items():
  image_index=cmd.index('sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7');cmd[image_index:image_index]=['-e',key+'='+value]
 cmd+=['--random-seed','0',*extra]
 (R/f'launch-{name}.json').write_text(json.dumps(cmd,indent=2))
 run(cmd);run(['docker','start',name]);return name

def stop(names):
 for name in names:
  with (R/(name+'-final.log')).open('x') as f:subprocess.run(['docker','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True)
 # All members are stopped together, including both sides of a PD pair.
 run(['docker','stop',*names])
def compare(name,stage,label,port,top,adapter='real_r16',limit=30,nt=256,rounds=4,batch=30,prefill=None,bootstrap=None):
 out=f'{top}-{label}-{stage}-r30'
 args=['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_frozen.py',stage,'--url',f'http://127.0.0.1:{port}','--topology',top+'-r30','--provenance',W+'/source-host-r50.json','--output',W+'/'+out]
 if stage=='reference':args+=['--corpus','/work/qwen38-mol-experiment/phase1/corpus.json' if adapter=='real_r16' else '/work/qwen38-mol-speed-20261019/corpus-mixed.json','--limit',str(limit),'--decode-tokens',str(nt),'--rounds',str(rounds),'--batch-size',str(batch)]
 else:args+=['--reference',W+f'/{top}-{label}-reference-r30','--adapter-path',ADAPTERS+adapter]
 if prefill is not None:args+=['--prefill-url',f'http://127.0.0.1:{prefill}','--bootstrap-port',str(bootstrap)]
 run(args,out+'.log');print('completed',out,flush=True)
 return json.loads((R/out/'summary.json').read_text())
def perf(name,port,top,side):
 out=f'perf-{top}-{side}-r30';cmd=['docker','exec',name,'python3',W+'/perf_endpoint_r29.py','--url',f'http://127.0.0.1:{port}','--output',W+'/'+out,'--profile']
 if side=='d':cmd+=['--adapter-path',ADAPTERS+'real_r16']
 run(cmd,out+'.log');print('completed',out,flush=True)
def compare_perf(top):
 left=[json.loads(s) for s in (R/f'perf-{top}-d-r30/requests.jsonl').read_text().splitlines()]
 right={r['label']:r for r in [json.loads(s) for s in (R/f'perf-{top}-m-r30/requests.jsonl').read_text().splitlines()]}
 assert len(left)==len(right)==636
 wrong=[r['label'] for r in left if r['output_ids']!=right[r['label']]['output_ids']]
 result={'requests_per_side':636,'outputs_equal':not wrong,'mismatched_labels':wrong}
 (R/f'perf-{top}-comparison-r30.json').write_text(json.dumps(result,indent=2));return result
