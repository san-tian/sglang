"""Near-pool resident KV check after restoring the 16-slot primary pair."""
import json,subprocess,threading,time,traceback
from pathlib import Path
import measure_1p1d_r160 as m
R=m.R
try:
 while not (R/'1p1d-slot-sweep-finished-r169.json').exists():
  assert not (R/'1p1d-slot-sweep-failed-r169.json').exists()
  time.sleep(3)
 m.OUT=R/'capacity-1p1d-million-resident-r182';m.OUT.mkdir(exist_ok=False);m.STOP=threading.Event()
 corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text());m.FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
 m.save('server-info.json',{role:m.call(url,'/get_server_info') for role,url in [('prefill',m.P),('decode',m.D)]})
 monitor=threading.Thread(target=m.monitor);monitor.start()
 try:
  m.batch(32768,2048,['main']*32,'capacity-in32768-c32')
 finally:m.STOP.set();monitor.join()
 m.save('complete.json',{'passed':True,'submitted_prompt_tokens':1048576,'requests':32,'input_tokens_each':32768,'output_tokens_each':2048,'scope':'Resident KV stress; not a single million-token context or long-context task quality'})
 subprocess.run(['python3',str(R/'audit_1p1d_kv_r183.py')],check=True)
except BaseException:
 with (R/'1p1d-KV-stress-failed-r182.json').open('x') as f:json.dump({'traceback':traceback.format_exc(),'unix_s':time.time()},f,indent=2)
 raise
