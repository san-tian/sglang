"""Separate-process, separate-GPU NIXL transfer preflight with original bytes."""
import argparse,base64,hashlib,importlib.metadata,json,os,time
from pathlib import Path
import torch
from nixl._api import nixl_agent,nixl_agent_config,nixl_thread_sync_t
p=argparse.ArgumentParser();p.add_argument('role',choices=['source','destination']);a=p.parse_args()
R=Path('/evidence/nixl-two-process-r75');R.mkdir(exist_ok=True)
def write(name,obj):
    tmp=R/(name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.rename(R/name)
def wait(name):
    deadline=time.monotonic()+120
    while not (R/name).exists():
        assert time.monotonic()<deadline,(a.role,name,'timeout')
        time.sleep(.01)
    return json.loads((R/name).read_text())
assert torch.cuda.device_count()==1
torch.cuda.set_device(0)
agent=nixl_agent('qwen38-r75-'+a.role,nixl_agent_config(backends=[],num_threads=8 if a.role=='source' else 0,sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT))
agent.create_backend('UCX',{'num_threads':'8'} if a.role=='source' else {})
assert 'VRAM_SEG' in agent.get_backend_mem_types('UCX')
records=[]
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for n in (4096,1048576,8388608):
        data=(bytes(range(256))*((n+255)//256))[:n]
        tensor=torch.frombuffer(bytearray(data),dtype=torch.uint8).to('cuda:0') if a.role=='source' else torch.zeros(n,dtype=torch.uint8,device='cuda:0')
        torch.cuda.synchronize()
        reg=agent.register_memory(tensor,backends=['UCX'])
        write(f'{a.role}-{n}-ready.json',{'address':tensor.data_ptr(),'bytes':n,'device':0,'metadata':base64.b64encode(agent.get_agent_metadata()).decode()})
        remote_role='destination' if a.role=='source' else 'source'
        remote=wait(f'{remote_role}-{n}-ready.json')
        remote_name=agent.add_remote_agent(base64.b64decode(remote['metadata']))
        if a.role=='source':
            target=agent.get_xfer_descs([(remote['address'],n,remote['device'])],'VRAM')
            handle=agent.initialize_xfer('WRITE',agent.get_xfer_descs(tensor),target,remote_name,backends=['UCX'])
            started=time.monotonic()
            with torch.profiler.record_function('SEPARATE_PROCESS_NIXL_WRITE_'+str(n)):
                state=agent.transfer(handle)
                while state=='PROC':
                    assert time.monotonic()-started<60
                    state=agent.check_xfer_state(handle)
            elapsed=time.monotonic()-started
            assert state=='DONE',state
            queried=str(agent.query_xfer_backend(handle));agent.release_xfer_handle(handle)
            write(f'done-{n}.json',{'bytes':n,'state':state,'queried_backend':queried,'host_observed_elapsed_s':elapsed})
            ack=wait(f'ack-{n}.json');assert ack['sha256']==hashlib.sha256(data).hexdigest()
            records.append({**json.loads((R/f'done-{n}.json').read_text()),'target_sha256':ack['sha256']})
        else:
            done=wait(f'done-{n}.json');assert done['state']=='DONE'
            torch.cuda.synchronize()
            actual=bytes(tensor.cpu().tolist());assert actual==data
            ack={'bytes':n,'sha256':hashlib.sha256(actual).hexdigest()};write(f'ack-{n}.json',ack);records.append(ack)
        agent.remove_remote_agent(remote_name);agent.deregister_memory(reg)
trace=R/(a.role+'-trace.json');prof.export_chrome_trace(str(trace))
result={'passed':True,'role':a.role,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'nixl_version':importlib.metadata.version('nixl'),'gpu':torch.cuda.get_device_name(),
        'ucx_tls':os.environ['UCX_TLS'],'pid':os.getpid(),'pid_namespace':os.readlink('/proc/self/ns/pid'),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'transfers':records,'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'scope':'Separate one-GPU processes, original byte equality across 4KiB/1MiB/8MiB CUDA VRAM writes. Actual UCX logs and traces retained. Not model PD, KV/SSM correctness/checksum or 4P4D performance. Host timings not bandwidth.'}
write(a.role+'-summary.json',result);print(json.dumps(result),flush=True)
