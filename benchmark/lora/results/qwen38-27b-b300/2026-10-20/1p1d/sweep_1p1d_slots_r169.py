"""Measure 32 GPU slots / 64 registered adapters; restore 16-slot primary pair."""
import copy,json,shutil,subprocess,threading,time,traceback
from pathlib import Path
import measure_1p1d_r160 as m

def copy_cache(src,dst):
 import stat
 def skip(folder,names):
  return [n for n in names if not (stat.S_ISREG((Path(folder)/n).lstat().st_mode) or stat.S_ISDIR((Path(folder)/n).lstat().st_mode) or stat.S_ISLNK((Path(folder)/n).lstat().st_mode))]
 shutil.copytree(src,dst,symlinks=True,ignore=skip)

R=m.R;OUT=R/'measure-1p1d-slots32-r169'
DN=['qwen38-1p1d-prefill-r160','qwen38-1p1d-decode-r160']
NAMES=['qwen38-1p1d-slots32-prefill-r169','qwen38-1p1d-slots32-decode-r169']
def save(name,obj):
 with (R/name).open('x') as f:json.dump(obj,f,indent=2)
def stop(names,suffix):
 for n in names:
  with (R/f'{n}-{suffix}.log').open('x') as f:subprocess.run(['docker','logs',n],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['docker','stop','--timeout','30',*names],check=True)
def ready():
 for _ in range(900):
  try:
   for url in [m.P,m.D]:m.call(url,'/health',timeout=3)
   return
  except Exception:time.sleep(2)
 raise TimeoutError('slot sweep startup')
def main():
 while not (R/'1p1d-control-finished-r168.json').exists():
  assert not (R/'1p1d-control-failed-r168.json').exists()
  assert not (R/'measure-1p1d-r160/failed.json').exists()
  time.sleep(5)
 stop(DN,'before-slot-sweep-r169')
 try:
  for name,oldname in zip(NAMES,DN):
   cmd=json.loads((R/f'launch-{oldname}.json').read_text());cmd[cmd.index('--name')+1]=name
   cmd[cmd.index('--max-loras-per-batch')+1]='32';cmd[cmd.index('--max-loaded-loras')+1]='64'
   cache=R/'caches'/name;copy_cache(R/'caches'/oldname,cache)
   for i,v in enumerate(cmd):
    if v.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
   save(f'launch-{name}.json',cmd);subprocess.run(cmd,check=True)
  subprocess.run(['docker','start',*NAMES],check=True);ready()
  m.OUT=OUT;OUT.mkdir(exist_ok=False);m.STOP=threading.Event()
  corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text())
  m.FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
  m.save('server-info.json',{role:m.call(url,'/get_server_info') for role,url in [('prefill',m.P),('decode',m.D)]})
  m.save('placement.json',{'gpu_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),'inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),'containers':json.loads(subprocess.check_output(['docker','inspect',*NAMES],text=True))})
  # Reuse the identical 32 scale variants from the first suite, add 32 distinct scales.
  scaled=json.loads((R/'measure-1p1d-r160/scaled-adapters.json').read_text())
  src=Path('/opt/dlami/nvme/qwen38-mol-experiment/adapters/real_r16')
  dest=R/'adapters-extra-r169';dest.mkdir(exist_ok=False)
  cfg=json.loads((src/'adapter_config.json').read_text())
  for i in range(32,64):
   folder=dest/f'scaled-{i:02d}';folder.mkdir();c=dict(cfg);c['lora_alpha']=float(cfg['lora_alpha'])*(i+1)/32
   (folder/'adapter_config.json').write_text(json.dumps(c))
   for f in src.iterdir():
    if f.name!='adapter_config.json':(folder/f.name).symlink_to(f)
   scaled.append({'name':f'slot-{i:02d}','path':str(folder).replace('/opt/dlami/nvme','/work',1),'alpha':c['lora_alpha'],'kind':'synthetic scale variant'})
  m.save('scaled-adapters.json',scaled)
  for x in scaled:m.control('/load_lora_adapter',{'lora_name':x['name'],'lora_path':x['path']})
  m.save('registered-models.json',{role:m.call(url,'/v1/models') for role,url in [('prefill',m.P),('decode',m.D)]})
  refs=[]
  for i in [0,31,63]:
   m.control('/flush_cache',{})
   refs.append(m.stream(m.prompt(128,i),16,scaled[i]['name'],f'slot-reference-{i}',logprobs=True))
  m.save('slot-reference-outputs.json',refs)
  for count in [1,16,32,64]:
   for repeat in range(2):m.batch(128,32,[x['name'] for x in scaled[:count]],f'slots-n{count}-r{repeat}',True)
  m.batch(128,32,[None]+[x['name'] for x in scaled[:32]],'slots-base-plus32',True)
  m.save('complete.json',{'passed':True,'GPU_slots_per_role':32,'registered_adapters_per_role':64,'full_goal_complete':False,'scope':'Capacity operating point, not hardware maximum. Distinct alpha scales of one real checkpoint.'})
 finally:
  stop(NAMES,'finished-r169')
  subprocess.run(['docker','start',*DN],check=True)
 ready();m.OUT=R/'primary-restored-r169';m.OUT.mkdir(exist_ok=False)
 m.control('/load_lora_adapter',{'lora_name':'main','lora_path':m.AD})
 m.batch(128,16,['main'],'restored-smoke')
 save('1p1d-slot-sweep-finished-r169.json',{'passed':True,'primary_dynamic_restored':True,'primary_GPU_slots':16,'sweep_GPU_slots':32,'sweep_registered_adapters':64,'unix_s':time.time(),'full_goal_complete':False})
if __name__=='__main__':
 try:main()
 except BaseException:
  save('1p1d-slot-sweep-failed-r169.json',{'traceback':traceback.format_exc(),'unix_s':time.time()})
  raise
