"""Sequential independent merged control, then restore the primary dynamic pair."""
import hashlib,json,shutil,subprocess,threading,time,traceback
from pathlib import Path
import measure_1p1d_r160 as m
R=m.R;W='/work/qwen38-mol-multigpu-20261019';CLIENT=W+'/client-r33'
DN=['qwen38-1p1d-prefill-r160','qwen38-1p1d-decode-r160']
MN=['qwen38-1p1d-merged-prefill-r161','qwen38-1p1d-merged-decode-r161']
OUT=R/'measure-1p1d-merged-r161'
def record(name,obj):
 with (R/name).open('x') as f:json.dump(obj,f,indent=2)
def waitready():
 for _ in range(900):
  try:
   for u in [m.P,m.D]:m.call(u,'/health',timeout=3)
   return
  except Exception:time.sleep(2)
 raise TimeoutError('pair startup')
def logstop(names,suffix):
 for name in names:
  with (R/f'{name}-{suffix}.log').open('x') as f:subprocess.run(['docker','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['docker','stop','--timeout','30',*names],check=True)
def precision(stage,name):
 cmd=['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_frozen.py',stage,'--url',m.D,'--prefill-url',m.P,'--bootstrap-port','31810','--topology','physical-1p1d-r161','--provenance',W+'/source-1p1d-r161.json','--output',W+f'/precision-1p1d-{stage}-r161']
 if stage=='reference':cmd+=['--corpus','/work/qwen38-mol-experiment/phase1/corpus.json','--limit','8','--decode-tokens','32','--rounds','2','--batch-size','8']
 else:cmd+=['--reference',W+'/precision-1p1d-reference-r161','--adapter-path',m.AD]
 with (R/f'precision-1p1d-{stage}-r161.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return json.loads((R/f'precision-1p1d-{stage}-r161/summary.json').read_text())
def main():
 while not (R/'measure-1p1d-r160/complete.json').exists():
  assert not (R/'measure-1p1d-r160/failed.json').exists(), 'Primary measurement failed; preserve pair for diagnosis'
  time.sleep(5)
 record('1p1d-merged-control-started-r161.json',{'unix_s':time.time(),'primary_complete':True})
 logstop(DN,'before-merge-r161')
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
 try:
  for role,name,dname in zip(['prefill','decode'],MN,DN):
   cmd=json.loads((R/f'launch-{dname}.json').read_text())
   cmd[cmd.index('--name')+1]=name
   cmd[cmd.index('--model-path')+1]='/work/qwen38-mol-development-20261019/merged/real_r16'
   cache=R/'caches'/name
   shutil.copytree(R/'caches'/dname,cache)
   for i,value in enumerate(cmd):
    if value.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
   lo=cmd.index('--enable-lora');hi=cmd.index('--random-seed');cmd=cmd[:lo]+cmd[hi:]
   # Match the measured dynamic tensor capacity rather than spending freed LoRA memory on extra KV.
   cmd+=['--max-total-tokens',str(json.loads((R/'measure-1p1d-r160/server-info.json').read_text())[role]['max_total_num_tokens'])]
   record(f'launch-{name}.json',cmd)
   subprocess.run(cmd,check=True)
  subprocess.run(['docker','start',*MN],check=True);waitready()
  m.OUT=OUT;OUT.mkdir(exist_ok=False);m.STOP=threading.Event()
  corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text())
  m.FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
  m.save('server-info.json',{role:m.call(url,'/get_server_info') for role,url in [('prefill',m.P),('decode',m.D)]})
  for nin,nout in [(128,128),(2048,256)]:
   for count in [1,32,64]:
    m.batch(nin,nout,[None]*count,f'warm-in{nin}-c{count}')
    for repeat in range(3):m.batch(nin,nout,[None]*count,f'speed-in{nin}-c{count}-r{repeat}')
  ref=precision('reference',MN[1])
  m.save('complete.json',{'passed':True,'scope':'Independent offline merged 1P1D, same physical GPUs and KV tensor capacities','precision_reference':ref,'full_goal_complete':False})
 finally:
  logstop(MN,'finished-r161')
  subprocess.run(['docker','start',*DN],check=True)
 waitready()
 dyn=precision('dynamic',DN[1])
 assert all(v['passes_logprob_gate'] for v in dyn['scores'].values()),dyn
 m.OUT=R/'primary-restored-r161';m.OUT.mkdir(exist_ok=False)
 m.control('/load_lora_adapter',{'lora_name':'main','lora_path':m.AD})
 m.batch(128,16,['main'],'restored-smoke')
 record('1p1d-control-finished-r161.json',{'passed':True,'primary_dynamic_restored':True,'scores':dyn,'full_goal_complete':False,'unix_s':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  record('1p1d-control-failed-r161.json',{'traceback':traceback.format_exc(),'unix_s':time.time()})
  raise
