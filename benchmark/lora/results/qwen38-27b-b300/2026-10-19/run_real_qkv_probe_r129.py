"""Use the rejected candidate only to localize its full-model discrepancy."""
from host_common_r118 import *

name='qwen38-dcp8x2-qkv-probe-r129'
port=30706
root=R/'real-qkv-r129'
assert not root.exists()
assert not Path('/proc/1264524').exists()
assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False
assert not (R/'eight-gpu-recovery-complete-r118.json').exists()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
assert not apps.strip(),apps
root.mkdir()
assert not json.loads(subprocess.check_output(['docker','inspect','qwen38-dcp8x2-qkv-probe-r127']))[0]['State']['Running']
source=R/'diagnostic-r129/sitecustomize.py'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
(root/'source.json').write_text(json.dumps({'runtime_image':'sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878',
    'diagnostic_hook_sha256':source_hash,'host_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'failed_report_sha256':hashlib.sha256((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_bytes()).hexdigest(),
    'purpose':'Locate the numerical failure with real model KV and attention-only oracles. No speed or acceptance claim.'},indent=2))
launch(name,list(range(8)),8,port,MERGES+'real_r16',extra=['--dcp-size','2','--dcp-comm-backend','a2a'],
       env={'PYTHONPATH':W+'/diagnostic-r129:'+CLIENT,'MOL_DCP_DIAGNOSTIC_ROOT_R129':W+'/real-qkv-r129'})
def request(path,payload):
 data=json.dumps(payload).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as f:return json.load(f)
try:
 health(port,name)
 import watch_actual_placement_r92 as observer
 observer.IMAGE='sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878'
 mapping={s.split(',')[1].strip():int(s.split(',')[0]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()}
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
 apps=[{'pid':int(v[0]),'gpu_uuid':v[1].strip(),'memory_mib':v[2].strip()} for v in [s.split(',') for s in raw.splitlines()] if len(v)==3]
 info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
 top=subprocess.check_output(['docker','top',name,'-eo','pid,ppid,args'],text=True)
 row=observer.inspect_placement(name,mapping,apps,info,top)
 assert row['placement_matched'] and len({p['pid'] for p in row['scheduler_gpu_processes']})==8
 (root/'placement.json').write_text(json.dumps(row,indent=2))
 corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-experiment/phase1/corpus.json').read_text())
 for label,cases in [('short',[corpus[0]]),('long',[next(x for x in corpus if x['id']=='long_prefix-1')]),('batch',corpus)]:
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/flush_cache',timeout=30) as f:f.read()
  (root/'enabled.json').write_text(json.dumps({'label':label}))
  payload={'input_ids':[x['input_ids'] for x in cases],
           'sampling_params':{'temperature':0,'max_new_tokens':1},'return_logprob':True,
           'logprob_start_len':0,'top_logprobs_num':1}
  result=request('/generate',payload)
  (root/f'{label}-response.json').write_text(json.dumps(result))
  (root/'enabled.json').unlink()
  print('completed real-QKV phase',label,flush=True)
finally:
 stop([name])
files=list(root.glob('*-rank*-layer*.json'))
assert files and len(files)%8==0
(root/'capture-finished.json').write_text(json.dumps({'oracle_pending':True,'diagnostic_captured':True,'records':len(files),
    'full_goal_complete':False,'numerical_failure_fixed':False,'time_unix':time.time()}))
