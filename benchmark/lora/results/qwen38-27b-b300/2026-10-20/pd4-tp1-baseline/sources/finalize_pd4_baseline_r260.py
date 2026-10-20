"""Capture end state and verify all eight workers use the same sealed engine overlay."""
import concurrent.futures as cf,gzip,hashlib,json,subprocess,time,urllib.request,uuid
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253';F=E/'final-state-r260';F.mkdir(exist_ok=False)
suite=json.loads((E/'baseline-complete-r264.json').read_text());assert suite['passed'],suite
assert json.loads((E/'residency-paced/complete.json').read_text())['passed']
assert json.loads((E/'slots-precision-r266/complete.json').read_text())['passed']
workers=json.loads((E/'topology.json').read_text())['workers']
def save(n,v):(F/n).write_text(json.dumps(v,indent=2))
expected=json.loads((R/'fp8-mtp-r228/runtime-overlay-audit-r248.json').read_text())['actual_sha256']
code="import hashlib,json,sglang;from pathlib import Path;p=Path(sglang.__file__).parent;ns="+repr(list(expected))+";print(json.dumps({'package':str(p),'files':{n:hashlib.sha256((p/n.removeprefix('python/sglang/')).read_bytes()).hexdigest() for n in ns}}))"
def worker(w):
 name=w['name'];actual=json.loads(subprocess.check_output(['docker','exec',name,'python3','-c',code],text=True))
 assert actual['files']==expected,(name,actual)
 with urllib.request.urlopen(w['url']+'/get_server_info',timeout=10) as f:info=json.load(f)
 with urllib.request.urlopen(w['url']+'/health',timeout=10) as f:health=f.status
 with urllib.request.urlopen(w['url']+'/v1/models',timeout=10) as f:models=json.load(f)
 assert any(x['id']=='main' for x in models['data']),models
 assert info['tp_size']==1 and info['context_length']==262144 and info['max_running_requests']==64
 assert info['kv_cache_dtype']=='fp8_e4m3' and info['speculative_algorithm']=='EAGLE'
 assert info['max_loras_per_batch']==16 and info['max_loaded_loras']==32
 assert info['mamba_radix_cache_strategy']=='extra_buffer'
 with gzip.open(F/(name+'.log.gz'),'wb') as f:f.write(subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT))
 return {'worker':w,'health_status':health,'models':models,'server_info':info,'overlay':actual}
with cf.ThreadPoolExecutor(max_workers=8) as pool:states=list(pool.map(worker,workers))
save('workers.json',states)
probes=[]
for i in range(4):
 p,d=workers[i],workers[i+4]
 q={'input_ids':[1000+j for j in range(16)],'stream':False,'lora_path':'main','sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True},'bootstrap_host':'127.0.0.1','bootstrap_port':p['bootstrap_port'],'bootstrap_room':uuid.uuid4().int&((1<<63)-1)}
 def request(w):
  with urllib.request.urlopen(urllib.request.Request(w['url']+'/generate',json.dumps(q).encode(),headers={'Content-Type':'application/json'}),timeout=30) as f:return json.load(f)
 start=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=2) as pool:
  a,b=list(pool.map(request,[p,d]))
 elapsed=time.monotonic()-start
 assert len(b['output_ids'])==8 and a['output_ids']==b['output_ids'][:1] and elapsed<3,(i,elapsed,a,b)
 probes.append({'p':p['name'],'d':d['name'],'elapsed_s':elapsed,'prefill':a,'decode':b})
save('post-reset-generation.json',probes)
save('runtime-overlay-audit.json',{'expected_fork_revision':'0116d3a12bc893ae43d8dbfd594280eccdc8467c','files_per_worker':26,'workers':8,'all_208_hashes_match':True,'actual':{s['worker']['name']:s['overlay'] for s in states}})
save('inventory.json',{'unix_s':time.time(),'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),'containers':subprocess.check_output(['docker','ps','--format','{{.Names}} {{.Status}}'],text=True)})
save('complete.json',{'unix_s':time.time(),'workers_healthy':8,'all_main_adapters_present':True,'all_208_overlay_hashes_match':True,'baseline_accuracy_accepted':False})
print('FINAL_STATE_VERIFIED',flush=True)
