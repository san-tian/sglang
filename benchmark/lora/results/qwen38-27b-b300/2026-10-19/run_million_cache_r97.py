"""Full six-arm million-context cache controls after existing isolated GPU lease."""
from host_common_r90 import *
import threading,traceback
IMAGE='sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7'
source=json.loads((R/'source-milli-cache-r97.json').read_text())
for n,h in source['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
corpus=R.parent/'qwen38-mol-speed-20261019/corpus-long-yarn4-r26.json'
assert hashlib.sha256(corpus.read_bytes()).hexdigest()==source['corpus_sha256']
assert not (R/'million-cache-finished-r97.json').exists()
waitleases(['isolated-speed-finished-r69.json','cached-topologies-finished-r50.json','pd4-precision-perf-complete-r33.json','eight-gpu-suite-complete-r31.json'],list(range(8)),timeout=172800)
assert json.loads((R/'isolated-speed-finished-r69.json').read_text())['passed']
stop_event=threading.Event();monitor_errors=[];monitor_times=[];owned=None
result={'passed':False,'scope':source['scope']}
def monitor():
    with (R/'million-cache-gpu-observations-r97.jsonl').open('x') as f:
        while not stop_event.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],text=True)
                rows=[[v.strip() for v in line.split(',')] for line in raw.splitlines()];assert len(rows)==8
                unexpected=[v for v in rows if int(v[0])!=2 and (int(v[1])>=2000 or int(v[2])!=0)]
                row={'unix_s':time.time(),'rows':rows,'unexpected_active_gpus':unexpected}
                if unexpected:monitor_errors.append(row)
            except BaseException:row={'unix_s':time.time(),'error':traceback.format_exc()};monitor_errors.append(row)
            f.write(json.dumps(row)+'\n');f.flush();monitor_times.append(row['unix_s']);stop_event.wait(1)
thread=threading.Thread(target=monitor);thread.start()
def release():
    global owned
    if owned:
        name,owned=owned,None
        if subprocess.run(['docker','inspect',name],capture_output=True).returncode==0:stop([name])
try:
    for index,side in enumerate(['nocache','cached','m','base','cached','nocache']):
        assert not monitor_errors,monitor_errors[-1:]
        assert shutil.disk_usage('/').free>4*1024**3
        owned=f'qwen38-million-{index}-{side}-r97';model=MERGES+'real_r16' if side=='m' else BASE+'/' if side=='base' else BASE
        flags=['--context-length','1000000','--max-total-tokens','1001024','--chunked-prefill-size','8192','--json-model-override-args',json.dumps(source['rope_override'])]
        if side=='cached':flags+=['--lora-merged-bf16-cache-size-mb','36864']
        launch(owned,[2],1,31500,model,memory=.8,mamba=16,requests=1,extra=flags,env={'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN':'1'})
        try:
            health(31500,owned)
            actual=json.loads(subprocess.check_output(['docker','inspect',owned]))[0];assert actual['Image']==IMAGE
            (R/(owned+'-actual-config.json')).write_text(json.dumps({'image':actual['Image'],'env':actual['Config']['Env'],'mounts':actual['Mounts']},indent=2))
            out=f'perf-million-{index}-{side}-r97';cmd=['docker','exec',owned,'python3',W+'/perf_million_cache_r97.py','--url','http://127.0.0.1:31500','--corpus','/work/qwen38-mol-speed-20261019/corpus-long-yarn4-r26.json','--output',W+'/'+out]
            if side in ['cached','nocache']:cmd+=['--adapter-path',ADAPTERS+'real_r16']
            run(cmd,out+'.log');assert not monitor_errors,monitor_errors[-1:]
        finally:release()
        print('completed',index,side,flush=True)
    terminal_time=time.time();deadline=time.monotonic()+15
    while not monitor_times or monitor_times[-1]<terminal_time:
        assert thread.is_alive() and not monitor_errors and time.monotonic()<deadline
        time.sleep(.1)
    stop_event.set();thread.join(30);assert not thread.is_alive() and not monitor_errors
    run(['python3',str(R/'audit_million_cache_r97.py')],'million-cache-audit-r97.log')
    result.update(passed=True,native_requests=36,warmups=12,artifact='million-cache-comparison-r97.json')
except BaseException:result['error']=traceback.format_exc();raise
finally:
    try:release()
    finally:
        stop_event.set();thread.join(30);result['monitor_errors']=monitor_errors
        with (R/'million-cache-finished-r97.json').open('x') as f:json.dump(result,f,indent=2)
