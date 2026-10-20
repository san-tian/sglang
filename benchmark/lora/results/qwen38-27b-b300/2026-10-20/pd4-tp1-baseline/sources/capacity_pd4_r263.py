"""Complete capacity checks with the experimentally discovered four-token MTP reserve."""
import importlib.util,json,sys,threading,time,traceback,urllib.error
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
for _ in range(1800):
 if (E/'suite-finished.json').exists():break
 time.sleep(1)
else:raise TimeoutError('original suite')
s=json.loads((E/'suite-finished.json').read_text())
assert all((E/p/'complete.json').exists() for p in ['smoke','speed','natural','slots']),s
assert (E/'capacity/failed.json').exists()
sys.argv=['measure_pd4_baseline_r254.py','capacity-r263']
spec=importlib.util.spec_from_file_location('pd4_native_capacity',R/'measure_pd4_baseline_r254.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mon=threading.Thread(target=m.monitor)
try:
 m.save('server-info.json',{w['name']:m.call(w['url'],'/get_server_info') for w in m.WORKERS});mon.start()
 for nin in [32768,131072,262012]:m.batch(nin,128,['main']*4,f'capacity-in{nin}-c4')
 for c in [32,64]:m.batch(131072,512,['main']*c,f'capacity-in131072-c{c}')
 rejected=[]
 for w in m.WORKERS:
  q={'input_ids':m.prompt(262013,0),'stream':False,'sampling_params':{'temperature':0,'max_new_tokens':128,'ignore_eos':True},'lora_path':'main'}
  try:
   v=m.call(w['url'],'/generate',q,timeout=20)
   assert 'error' in v or v.get('meta_info',{}).get('finish_reason',{}).get('type')=='abort',(w['name'],v)
   rejected.append({'worker':w['name'],'response':v})
  except urllib.error.HTTPError as ex:
   assert ex.code==400,(w['name'],ex);rejected.append({'worker':w['name'],'status':400,'raw_error_in':'http-errors.jsonl'})
 m.save('boundary-rejections.json',{'raw_input':262013,'requested_output':128,'mtp_reserved':4,'validated_total':262145,'responses':rejected})
 m.save('complete.json',{'passed':True,'requests':108,'configured_context':262144,'maximum_tested_user_input_plus_output':262140,'mtp_reserved_tokens':4,'all_8_workers_reject_one_token_over_budget':True,'original_attempt_preserved':'capacity/failed.json'})
 (E/'baseline-complete-r264.json').write_text(json.dumps({'passed':True,'normal_phases':['smoke','speed','natural','capacity-r263','slots'],'original_suite':'suite-finished.json','original_capacity_attempt_retained':True,'resolved_capacity':'capacity-r263/complete.json','unix_s':time.time()},indent=2))
except BaseException:
 m.save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()});raise
finally:m.STOP.set();mon.join(30);m.PPOOL.shutdown(wait=True)
