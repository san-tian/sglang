"""Audit four real GPU transport controls; do not infer data path from TLS flags."""
import collections,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
IMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result={'passed':True,'runtime_image':IMAGE,'physical_gpus':[0,5],'controls':{}}
    for run,shared_pid,shared_devices in [(74,False,False),(75,True,False),(76,True,True),(77,False,True)]:
        dirname=f'nixl-two-process-r{run}';p=R/dirname;proof=read(R/f'nixl-two-process-launch-r{run}.json')
        sources=read(p/'source-summary.json');destination=read(p/'destination-summary.json')
        expected={n:hashlib.sha256((bytes(range(256))*((n+255)//256))[:n]).hexdigest() for n in (4096,1048576,8388608)}
        assert sources['passed'] and destination['passed']
        assert {x['bytes']:x['target_sha256'] for x in sources['transfers']}==expected
        assert {x['bytes']:x['sha256'] for x in destination['transfers']}==expected
        assert all(x['state']=='DONE' and x['queried_backend']=='UCX' for x in sources['transfers'])
        actual={};traces={};files={};logs={}
        for role in ('source','destination'):
            name=f'qwen38-nixl-{role}-r{run}'
            inspect=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            assert inspect['Image']==IMAGE and not inspect['State']['Running'] and not inspect['State']['OOMKilled'] and inspect['State']['ExitCode']==0
            h=inspect['HostConfig'];assert h['IpcMode']==h['NetworkMode']=='host'
            assert h['PidMode']==('host' if shared_pid else '')
            devices=h['DeviceRequests'][0]['DeviceIDs'];assert devices==(['0','5'] if shared_devices else (['0'] if role=='source' else ['5']))
            actual[role]={'image':inspect['Image'],'devices':devices,'pid_mode':h['PidMode'],'exit_code':inspect['State']['ExitCode'],'OOMKilled':False}
            raw=read(p/(role+'-trace.json'));assert sha(p/(role+'-trace.json'))==read(p/(role+'-summary.json'))['trace_sha256']
            copies=collections.Counter(e['name'] for e in raw['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='gpu_memcpy')
            traces[role]={'gpu_memcpy_counts':dict(copies),'sha256':sha(p/(role+'-trace.json'))}
            log=R/(name+'.log');text=log.read_text();logs[role]={'sha256':sha(log),'selected_device_cuda_ipc_logged':'device(cuda_ipc/cuda)' in text}
            for n in ['summary.json','trace.json']:
                f=p/(role+'-'+n);files[f'{role}-{n}']=sha(f)
        direct=(run==76)
        if direct:
            assert traces['source']['gpu_memcpy_counts']=={'Memcpy HtoD (Pageable -> Device)':3,'Memcpy PtoP (Device -> Device)':3}
            assert traces['destination']['gpu_memcpy_counts']=={'Memcpy DtoH (Device -> Pageable)':3}
            assert all(x['selected_device_cuda_ipc_logged'] for x in logs.values())
        else:
            assert traces['source']['gpu_memcpy_counts']=={'Memcpy HtoD (Pageable -> Device)':3,'Memcpy DtoH (Device -> Pageable)':1157}
            assert traces['destination']['gpu_memcpy_counts']=={'Memcpy HtoD (Pageable -> Device)':1157,'Memcpy DtoH (Device -> Pageable)':3}
        result['controls'][str(run)]={'passed':True,'shared_host_pid_namespace':shared_pid,'both_devices_visible_to_both_roles':shared_devices,'direct_PtoP_observed':direct,'actual_containers':actual,'traces':traces,'logs':logs,'target_hashes':expected,'raw_files_sha256':files,'launch_proof_sha256':sha(R/f'nixl-two-process-launch-r{run}.json')}
    result.update(audit_source_sha256=sha(Path(__file__)),scope='Four actual separate-process NIXL/UCX controls on GPU0/5 with4KiB/1MiB/8MiB exact-byte writes. Only shared host PID plus both GPU devices visible yielded3 direct PtoP copies and zero staging copies; other three controls each show1157 source DtoH and1157 target HtoD copies. CUDA IPC device-lane logging alone was insufficient in R75. Initial HtoD and final DtoH are input setup/verification. Tiny profiled one-shot transfers are not bandwidth or model PD benchmarks; actual4P4D KV/SSM, precision and performance remain pending.')
    out=R/'nixl-transport-controls-audit-r78.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'controls':4,'transfers':12,'direct_PtoP_control':76,'staged_controls':[74,75,77]}))
if __name__=='__main__':main()
