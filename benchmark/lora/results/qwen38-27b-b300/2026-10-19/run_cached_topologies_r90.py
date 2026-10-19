"""Current R34 cached TP4/attention-DP4 controls after native 4P4D finishes.

Each topology runs alone on GPUs0..3 while all other GPUs remain idle. Never
change the still-running R28 topology/PD experiments to fit this candidate.
"""
from host_common_r90 import *
import threading
import traceback

manifest = json.loads((R / 'source-host-r90.json').read_text())
for file, sha in manifest['source_files'].items():
    assert hashlib.sha256((R / file).read_bytes()).hexdigest() == sha, file
waitleases(['pd4-precision-perf-complete-r33.json', 'eight-gpu-suite-complete-r31.json'], list(range(8)), timeout=172800)
end_monitor = threading.Event()
monitor_errors = []
active_gpus = {0,1,2,3}
owned = None
observations = []
observation_lock = threading.Lock()

def observe():
    with (R / 'cached-topologies-gpu-observations-r50.jsonl').open('x') as out:
        while not end_monitor.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits'], text=True)
                rows = []
                for line in raw.splitlines():
                    v = [x.strip() for x in line.split(',')]
                    rows.append({'index': int(v[0]), 'memory_mib': int(v[1]), 'utilization_pct': int(v[2]), 'temperature_c': v[3], 'power_w': v[4], 'sm_clock_mhz': v[5]})
                allowed = active_gpus.copy()
                wrong = [row for row in rows if row['index'] not in allowed and (row['memory_mib'] >= 2000 or row['utilization_pct'] != 0)]
                record = {'unix_s': time.time(), 'active_gpu_allowlist': sorted(allowed), 'gpus': rows, 'unexpected': wrong}
                out.write(json.dumps(record) + '\n'); out.flush()
                with observation_lock: observations.append(record)
                if wrong: monitor_errors.append(record)
            except Exception:
                monitor_errors.append({'unix_s': time.time(), 'error': traceback.format_exc()})
            end_monitor.wait(1)

thread = threading.Thread(target=observe); thread.start()
result = {'passed': False, 'scope': 'Current R34 runtime, fixed KV65536/SSM256, TP4 and attention-DP4 on the same GPUs0..3 sequentially; M, cached D, uncached D and pure-base each636 natural requests/topology. Main240/GDN8 and mixed48 gates run on cached D; uncached D has the natural-output comparison but no separate forced-score main matrix.'}
try:
    for top, dpa, budget, port in [('cache-tp4-r50', False, 10240, 31300), ('cache-dpa4-r50', True, 12288, 31302)]:
        assert not monitor_errors, monitor_errors[-1:]
        active_gpus = {0, 1, 2, 3}
        fixed = ['--max-total-tokens', '65536']
        for adapter, label in [('real_r16', 'full'), ('gdn_fused', 'gdn')]:
            name = f'qwen38-{top}-m-{adapter}'
            launch(name, [0,1,2,3], 4, port, MERGES + adapter, dpa=dpa, extra=fixed)
            owned = name
            try:
                health(port, name)
                compare(name, 'reference', label, port, top, adapter, **({} if label == 'full' else {'limit':4, 'nt':16, 'rounds':1, 'batch':4}))
                if label == 'full': perf(name, port, top, 'm')
            finally:
                stop([name]); owned = None
        name = f'qwen38-{top}-base'
        launch(name, [0,1,2,3], 4, port, BASE+'/', dpa=dpa, extra=fixed); owned = name
        try:
            health(port, name); perf(name, port, top, 'base')
        finally:
            stop([name]); owned = None
        name = f'qwen38-{top}-d'
        launch(name, [0,1,2,3], 4, port, BASE, dpa=dpa, extra=fixed+['--lora-merged-bf16-cache-size-mb', str(budget)]); owned = name
        try:
            health(port, name)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/get_server_info') as f: info=json.load(f)
            assert info['max_total_num_tokens']==65536 and info['max_mamba_cache_size']==256 and info['lora_merged_bf16_cache_size_mb']==budget, info
            for adapter, label in [('real_r16', 'full'), ('gdn_fused', 'gdn')]:
                s=compare(name,'dynamic',label,port,top,adapter)
                assert s['case_pairs']==(240 if label=='full' else 8) and all(v['passes_logprob_gate'] for v in s['scores'].values()),s
            adapters=[{'name':n,'path':ADAPTERS+a} for n,a in [('real','real_r16'),('zero','zero'),('gdn','gdn_fused')]]
            (R/f'{top}-mixed-adapters.json').write_text(json.dumps(adapters))
            run(['docker','exec','-e','PYTHONPATH='+CLIENT,name,'python3',CLIENT+'/compare_mixed.py','--url',f'http://127.0.0.1:{port}','--adapters',W+f'/{top}-mixed-adapters.json','--corpus','/work/qwen38-mol-speed-20261019/corpus-mixed.json','--capture-dir',W+f'/{top}-mixed-captures','--output',W+f'/{top}-mixed','--cases','4','--decode-tokens','16','--rounds','3'],f'{top}-mixed.log')
            perf(name,port,top,'d')
        finally:
            stop([name]); owned=None
        name=f'qwen38-{top}-nocache'
        launch(name,[0,1,2,3],4,port,BASE,dpa=dpa,extra=fixed);owned=name
        try:
            health(port,name)
            out=f'perf-{top}-nocache-r30'
            run(['docker','exec',name,'python3',W+'/perf_endpoint_r85.py','--url',f'http://127.0.0.1:{port}','--output',W+'/'+out,'--adapter-path',ADAPTERS+'real_r16','--profile'],out+'.log')
        finally:
            stop([name]);owned=None
        for side in ['m','d','base','nocache']:
            out=R/f'perf-{top}-{side}-r30'
            run(['python3',str(R/'analyze_profiles_r30.py'),str(out/'profile'),'--output',str(out/'kernel-summary.json')])
        assert not monitor_errors,monitor_errors[-1:]
        with observation_lock:
            (R/f'{top}-gpu-observation-snapshot.json').write_text(json.dumps(observations))
        run(['python3',str(R/'audit_cached_topology_r50.py'),top],f'{top}-audit.log')
        (R/f'{top}-complete.json').write_text(json.dumps({'passed':True,'cached_main_pairs':240,'cached_gdn_pairs':8,'mixed_requests':48,'natural_requests_per_side':636,'cache_budget_mib_per_rank':budget,'physical_gpus':[0,1,2,3],'scope':'Uncached D natural equivalence checked; independent main/GDN forced-score gate covers cached D. Other GPUs idle in sampled observations.'},indent=2))
        active_gpus=set()
        print('completed',top,flush=True)
    result.update(passed=True,topologies=['cache-tp4-r50','cache-dpa4-r50'],natural_requests=5088)
except Exception:
    result['error']=traceback.format_exc();raise
finally:
    if owned: stop([owned])
    end_monitor.set();thread.join(30)
    result['monitor_errors']=monitor_errors
    if monitor_errors:result['passed']=False
    (R/'cached-topologies-finished-r50.json').write_text(json.dumps(result,indent=2))
