"""PD decode residency larger than prefill pool; pacing is never a speed score."""
import json,sys,threading,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228'
for i in range(900):
 if (E/'slots-suite-complete-r240.json').exists():break
 time.sleep(1)
else:raise TimeoutError('prior suite')
sys.argv=['measure_fp8_r239.py','pd','residency-paced']
m=type(sys)('native_hold_fp8_r236');exec(compile((R/'measure_fp8_r239.py').read_text(),'measure_fp8_r239.py','exec'),m.__dict__)
DONE=threading.Event();GATE={'passed':False,'samples':[]};mon=gate=None
from audit_fp8_r235 import metric
def pace(value):
 response=m.call(m.TARGET,'/slow_down',{'forward_sleep_time':value});m.append('pacing-controls.jsonl',{'unix_s':time.time(),'decode_only':True,'forward_sleep_time':value,'response':response})
def monitor_gate():
 start=time.monotonic();qualified=[]
 try:
  while time.monotonic()-start<360 and not DONE.is_set():
   raw=m.call(m.TARGET,'/metrics',timeout=8);now=time.monotonic();metrics={k:metric(raw,k) for k in ['kv_used_tokens','num_running_reqs','decode_sum_seq_lens']}
   m.append('gate-metrics.jsonl',{'monotonic':now,'raw':raw,'metrics':metrics})
   if metrics.get('num_running_reqs')==32 and (metrics.get('decode_sum_seq_lens') or 0)>=32*131072:qualified.append({'monotonic':now,'metrics':metrics})
   if len(qualified)>=2:GATE.update(passed=True,samples=qualified);break
   DONE.wait(.5)
  GATE['seconds_until_gate_or_stop']=time.monotonic()-start
 finally:pace(None)
try:
 m.save('server-info.json',{r:m.call(u,'/get_server_info') for r,u in m.URLS.items()})
 assert all(any(x['id']=='main' for x in m.call(u,'/v1/models')['data']) for u in m.URLS.values())
 pace(2.0);mon=threading.Thread(target=m.monitor);mon.start();gate=threading.Thread(target=monitor_gate);gate.start()
 m.batch(131072,512,['main']*32,'capacity-paced-in131072-c32')
 DONE.set();gate.join(30)
 m.save('complete.json',{'passed':True,'completed_requests':32,'input_tokens_each':131072,'output_tokens_each':512,'residency_gate':GATE,'scope':'Artificially paced decode only at 2s/forward, reset at gate or 360s timeout. Not a speed score or task quality. Tests D-residency independently of the P-pool capacity.'})
except BaseException:
 m.save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()});raise
finally:
 DONE.set()
 try:pace(None)
 finally:
  m.STOP.set()
  if mon:mon.join(30)
  if gate:gate.join(30)
