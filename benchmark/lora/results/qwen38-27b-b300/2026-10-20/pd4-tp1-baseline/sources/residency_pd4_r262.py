"""Hold 32x128K requests on each D to verify aggregate residency; never a speed score."""
import concurrent.futures as cf,importlib.util,json,sys,threading,time,traceback
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
for _ in range(1800):
 if (E/'baseline-complete-r264.json').exists():break
 time.sleep(1)
else:raise TimeoutError('baseline suite')
assert json.loads((E/'baseline-complete-r264.json').read_text())['passed']
sys.argv=['measure_pd4_baseline_r254.py','residency-paced']
spec=importlib.util.spec_from_file_location('pd4_native',R/'measure_pd4_baseline_r254.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from audit_fp8_r247 import metric
DONE=threading.Event();GATE={'passed':False,'qualified_windows':[]};mon=gate=None
def pace(value):
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  rs=list(pool.map(lambda w:{'worker':w['name'],'response':m.call(w['url'],'/slow_down',{'forward_sleep_time':value},timeout=20)},m.DS))
 m.append('pacing-controls.jsonl',{'unix_s':time.time(),'forward_sleep_time':value,'responses':rs})
def watch():
 start=time.monotonic()
 try:
  with cf.ThreadPoolExecutor(max_workers=4) as pool:
   while time.monotonic()-start<600 and not DONE.is_set():
    def snap(w):
     t=time.monotonic();raw=m.call(w['url'],'/metrics',timeout=10)
     return {'worker':w['name'],'index':w['index'],'started_monotonic':t,'ended_monotonic':time.monotonic(),'raw':raw,'metrics':{k:metric(raw,k) for k in ['num_running_reqs','decode_sum_seq_lens','kv_used_tokens','max_total_num_tokens']}}
    ss=list(pool.map(snap,m.DS));span=max(s['ended_monotonic'] for s in ss)-min(s['started_monotonic'] for s in ss)
    row={'monotonic':time.monotonic(),'span_s':span,'samples':ss};m.append('gate-metrics.jsonl',row)
    if span<=1 and all(s['metrics']['num_running_reqs']==32 and (s['metrics']['decode_sum_seq_lens'] or 0)>=32*131072 for s in ss):GATE['qualified_windows'].append(row)
    if len(GATE['qualified_windows'])>=2:GATE['passed']=True;break
    DONE.wait(.5)
  GATE['seconds_until_gate_or_stop']=time.monotonic()-start
 finally:pace(None)
try:
 m.save('server-info.json',{w['name']:m.call(w['url'],'/get_server_info') for w in m.WORKERS})
 pace(4.0);mon=threading.Thread(target=m.monitor);mon.start();gate=threading.Thread(target=watch);gate.start()
 rows,g=m.batch(131072,512,['main']*128,'capacity-paced-in131072-c128')
 DONE.set();gate.join(30)
 m.save('complete.json',{'passed':GATE['passed'],'completed_requests':len(rows),'residency_gate':GATE,'scope':'4s per-forward artificial decode pacing, reset at gate or timeout; never a throughput score; concurrent polls <=1s span, not atomic global snapshot'})
 assert GATE['passed'],GATE
except BaseException:
 m.save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()});raise
finally:
 DONE.set()
 try:pace(None)
 finally:
  m.STOP.set()
  if mon:mon.join(30)
  if gate:gate.join(30)
  m.PPOOL.shutdown(wait=True)
