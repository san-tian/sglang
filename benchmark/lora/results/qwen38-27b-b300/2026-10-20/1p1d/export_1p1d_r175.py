"""Export only completed 1P1D cohorts, immutable raw captures, and their sources."""
import gzip,hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent
assert json.loads((R/'1p1d-audits-finished-r174.json').read_text())['passed']
assert json.loads((R/'1p1d-KV-observed-audit-r185.json').read_text())['passed']
import measure_1p1d_r160 as m
m.OUT=R/'primary-final-r186';m.OUT.mkdir(exist_ok=False)
corpus=json.loads(Path('/opt/dlami/nvme/qwen38-mol-merged-bf16-20261019/corpus-mixed.json').read_text());m.FILLER=max(corpus,key=lambda x:len(x['input_ids']))['input_ids']
m.batch(128,16,['main'],'post-stress-smoke')
D=R/'publish-1p1d-r175';D.mkdir(exist_ok=False)
SOURCES=['audit_1p1d_kv_observed_r185.py','stress_1p1d_kv_r182.py','audit_1p1d_kv_r183.py','launch_1p1d_r160.py','measure_1p1d_r160.py','merge_control_1p1d_r161.py','sweep_1p1d_slots_r163.py','audit_1p1d_r162.py','audit_1p1d_complete_r166.py','audit_1p1d_slots_r164.py','finalize_1p1d_r167.py','provide_adapter_path_alias_r165.py','merge_control_1p1d_r168.py','sweep_1p1d_slots_r169.py','audit_1p1d_r170.py','audit_1p1d_complete_r171.py','audit_1p1d_slots_r172.py','provide_adapter_path_alias_r173.py','finalize_1p1d_r174.py','start_recovery_1p1d_r168.py','export_1p1d_r175.py']
FILES=['1p1d-KV-stress-failed-r182.json','1p1d-KV-observed-audit-r185.json','source-1p1d-kv-r182.json','input-provenance-1p1d-r178.json','source-1p1d-r161.json','source-1p1d-r168.json','1p1d-main-audit-r171.json','1p1d-slot-sweep-audit-r172.json','1p1d-audits-finished-r174.json','1p1d-control-finished-r168.json','1p1d-slot-sweep-finished-r169.json','1p1d-control-failed-r161.json','1p1d-slot-sweep-failed-r163.json','recovery-processes-r168.json','priority-1p1d-reservation-r154.json']
for n in SOURCES+FILES:shutil.copyfile(R/n,D/n)
shutil.copyfile('/opt/dlami/nvme/qwen38-mol-development-20261019/merged/real_r16/merge-manifest.json',D/'reference-merge-manifest.json')
for pattern in ['launch-qwen38-1p1d-*.json','adapter-path-alias-*-r165.json','adapter-path-alias-*-r173.json']:
 for f in R.glob(pattern):shutil.copyfile(f,D/f.name)
logs=set()
for pattern in ['KV-stress-1p1d-host-r182.log','qwen38-1p1d-*-r160-before-merge-r168.log','qwen38-1p1d-slots32-*-r169-finished-r169.log','qwen38-1p1d-merged-*-r168-finished-r168.log','*1p1d-host-r160.log','*1p1d-host-r168.log','*1p1d-host-r169.log','1p1d-main-audit-r171.log','1p1d-slot-sweep-audit-r172.log','merge-control-1p1d-host-r161.log','finalize-1p1d-host-r174.log']:
 logs.update(R.glob(pattern))
with tarfile.open(D/'startup-and-control-logs.tgz','w:gz',compresslevel=6) as t:
 for f in sorted(logs):t.add(f,arcname=f.name,recursive=False)
folders=['primary-final-r186','capacity-1p1d-million-resident-r182','measure-1p1d-r160','measure-1p1d-merged-r168','measure-1p1d-slots32-r169','precision-1p1d-reference-r168','precision-1p1d-dynamic-r168','primary-restored-r168','primary-restored-r169']
raw_manifest={}
for name in folders:
 folder=R/name;assert folder.is_dir()
 for f in folder.rglob('*'):
  if f.is_file():raw_manifest[str(f.relative_to(R))]=hashlib.sha256(f.read_bytes()).hexdigest()
  assert not f.is_symlink(),f
 # No model weights, host caches or adapter weights are included.
 with tarfile.open(D/(name+'.tgz'),'w:gz',compresslevel=6) as t:t.add(folder,arcname=name)
 with (D/(name+'-metadata.json')).open('x') as f:
  json.dump({n:json.loads((folder/n).read_text()) for n in ['complete.json','summary.json','seal.json','server-info.json'] if (folder/n).exists()},f,indent=2)
# Audit the final pair with native inference already completed by R169 restore.
from measure_1p1d_r160 import call,P,D as URL_D
live={'unix_s':time.time(),'health':{'prefill':call(P,'/health'),'decode':call(URL_D,'/health')},'server_info':{'prefill':call(P,'/get_server_info'),'decode':call(URL_D,'/get_server_info')},'registered_models':{'prefill':call(P,'/v1/models'),'decode':call(URL_D,'/v1/models')},'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),'gpu_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),'restore':json.loads((R/'1p1d-slot-sweep-finished-r169.json').read_text())}
with (D/'final-live-state-r175.json').open('x') as f:json.dump(live,f,indent=2)
with (D/'raw-files-sha256.json').open('x') as f:json.dump(raw_manifest,f,indent=2)
manifest={'unix_s':time.time(),'scope':'Two physical B300 GPUs P TP1 + D TP1; BF16 eager merged_bf16; measured operating points, not hardware maxima. Original R161 harness failure retained; recovery R168/R169 completed. Old 8-GPU suite remains deferred.','files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.iterdir() if f.is_file()}}
with (D/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
for f in D.iterdir():assert f.stat().st_size<95*1024*1024,(f,f.stat().st_size)
print(json.dumps({'files':len(manifest['files_sha256']),'raw_files':len(raw_manifest),'total_MiB':sum(f.stat().st_size for f in D.iterdir())/1024**2,'destination':str(D)}))
