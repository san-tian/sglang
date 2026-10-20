import hashlib,json,subprocess,time
from pathlib import Path
r=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
source=json.loads((r/'source-1p1d-r161.json').read_text())
new=['merge_control_1p1d_r168.py','sweep_1p1d_slots_r169.py','audit_1p1d_r170.py','audit_1p1d_complete_r171.py','audit_1p1d_slots_r172.py','provide_adapter_path_alias_r173.py','finalize_1p1d_r174.py','start_recovery_1p1d_r168.py']
source['source_files_sha256'].update({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in new})
source['recovery_note']='R161 failed before merged model startup copying ephemeral socket files; R168 copies only regular files/directories/symlinks. R160 measurements unchanged. Failed run retained.'
with (r/'source-1p1d-r168.json').open('x') as f:json.dump(source,f,indent=2)
launched=[]
for script,log in [('provide_adapter_path_alias_r173.py','adapter-path-alias-host-r173.log'),('merge_control_1p1d_r168.py','merge-control-1p1d-host-r168.log'),('sweep_1p1d_slots_r169.py','slot-sweep-1p1d-host-r169.log'),('finalize_1p1d_r174.py','finalize-1p1d-host-r174.log')]:
 p=subprocess.Popen(['python3','-u',str(r/script)],stdout=(r/log).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
 launched.append({'pid':p.pid,'script':script})
with (r/'recovery-processes-r168.json').open('x') as f:json.dump({'unix_s':time.time(),'processes':launched},f,indent=2)
print(json.dumps(launched))
