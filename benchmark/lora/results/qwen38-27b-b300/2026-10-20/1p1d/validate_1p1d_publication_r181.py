"""Validate exported evidence, source seals, two-GPU placement and final state."""
import hashlib,json,tarfile
from pathlib import Path
D=Path('/home/dev/Deployment/qwen38-mol-sglang/benchmark/lora/results/qwen38-27b-b300/2026-10-20/1p1d')
def j(n):return json.loads((D/n).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(folder,name):
 with tarfile.open(D/(folder+'.tgz')) as t:return json.loads(t.extractfile(folder+'/'+name).read())
for name,h in j('manifest.json')['files_sha256'].items():assert sha((D/name).read_bytes())==h,name
for name in ['source-1p1d-r168.json','source-1p1d-kv-r182.json']:
 for n,h in j(name)['source_files_sha256'].items():assert sha((D/n).read_bytes())==h,n
expected=j('raw-files-sha256.json');seen={}
for tgz in D.glob('*.tgz'):
 if tgz.name=='startup-and-control-logs.tgz':continue
 with tarfile.open(tgz) as t:
  for m in t:
   assert not m.issym() and not m.islnk(),m.name
   if not m.isfile():continue
   assert m.name in expected,m.name
   seen[m.name]=sha(t.extractfile(m).read());assert seen[m.name]==expected[m.name],m.name
assert seen==expected
for name in ['1p1d-main-audit-r171.json','1p1d-slot-sweep-audit-r172.json','1p1d-KV-observed-audit-r185.json']:assert j(name)['passed']
place=raw('measure-1p1d-slots32-r169','placement.json')
uuid_index={s.split(',')[1].strip():int(s.split(',')[0]) for s in place['inventory'].splitlines()}
apps=[s.split(',') for s in place['gpu_apps'].splitlines() if s.strip()];heavy=[x for x in apps if float(x[2].strip().split()[0])>100000]
assert len(heavy)==2 and {uuid_index[x[1].strip()] for x in heavy}=={0,1}
for i,item in enumerate(place['containers']):
 assert item['HostConfig']['DeviceRequests'][0]['DeviceIDs']==['0','1']
 cmd=item['Config']['Cmd'];assert cmd[cmd.index('--base-gpu-id')+1]==str(i)
for folder in ['primary-restored-r168','primary-restored-r169']:
 data=raw(folder,'restored-smoke-i0.json');assert len(data['decode_events'][-1]['data']['output_ids'])==16
live=j('final-live-state-r175.json');assert live['restore']['primary_dynamic_restored']
for role,info in live['server_info'].items():
 cfg=info.get('server_args',info);assert cfg['max_loras_per_batch']==16 and cfg['max_loaded_loras']==32 and cfg['context_length']==262144
# All actual GPU apps after stress must still be on the two owned devices.
uuid_index={s.split(',')[1].strip():int(s.split(',')[0]) for s in live['gpu_inventory'].splitlines()}
assert {uuid_index[s.split(',')[1].strip()] for s in live['gpu_apps'].splitlines() if s.strip()}=={0,1}
observed=j('1p1d-KV-observed-audit-r185.json');assert observed['full32_residency_gate_passed'] is False and observed['original_full32_gate_failure_preserved']
assert observed['peak_active_sequence_sample']['metrics']['decode_sum_seq_lens']>1000000
print(json.dumps({'passed':True,'full32_stress_gate_passed':False,'exported_files_checked':len(j('manifest.json')['files_sha256']),'raw_files_checked':len(seen),'primary_and_sweep_physical_GPUs':[0,1],'primary_restored_GPU_slots':16,'other_six_GPUs_idle':True}))
