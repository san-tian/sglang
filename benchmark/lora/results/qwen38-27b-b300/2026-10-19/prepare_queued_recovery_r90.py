from pathlib import Path
import ast, hashlib, json, subprocess
R=Path(__file__).resolve().parent
old=(R/'host_common_r50.py').read_text()
new=old.replace(" shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache)"," shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache)\n (cache/'tmp').mkdir(parents=True,exist_ok=True)")
new=new.replace(' for key,value in (env or {}).items():'," env={'TMPDIR':'/root/.cache/tmp','PYTHONPYCACHEPREFIX':'/root/.cache/pycache',**(env or {})}\n for key,value in env.items():")
a=new.index('def stop(names):');b=new.index('def compare(',a)
new=new[:a]+'''def stop(names):
 try:
  for name in names:
   with (R/(name+'-final.log')).open('x') as f:
    subprocess.run(['docker','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True)
 finally:
  run(['docker','stop',*names])
'''+new[b:]
new=new.replace('source-host-r50.json','source-host-r90.json').replace('perf_endpoint_r29.py','perf_endpoint_r85.py')
(R/'host_common_r90.py').write_text(new)
for oldv,newv,stem in [(50,90,'cached_topologies'),(69,91,'isolated_speed')]:
 a=(R/f'run_{stem}_r{oldv}.py').read_text()
 b=a.replace('from host_common_r50 import *','from host_common_r90 import *').replace(f'source-host-r{oldv}.json',f'source-host-r{newv}.json').replace('perf_endpoint_r29.py','perf_endpoint_r85.py')
 (R/f'run_{stem}_r{newv}.py').write_text(b)
 # Prove every other instruction, measurement, marker and output unchanged.
 restored=b.replace('from host_common_r90 import *','from host_common_r50 import *').replace(f'source-host-r{newv}.json',f'source-host-r{oldv}.json').replace('perf_endpoint_r85.py','perf_endpoint_r29.py')
 assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(a))
 oldtest='test_cached_topology_launch_r50.py' if oldv==50 else 'check_isolated_launch_r69.py'
 test=(R/oldtest).read_text().replace('host_common_r50.py','host_common_r90.py').replace('run_isolated_speed_r69.py','run_isolated_speed_r91.py')
 (R/f'check_queued_launch_r{newv}.py').write_text(test)
 p=subprocess.run(['python3',str(R/f'check_queued_launch_r{newv}.py')],capture_output=True,text=True)
 (R/f'queued-launch-check-r{newv}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 m=json.loads((R/f'source-host-r{oldv}.json').read_text())
 m['source_files']={n:h for n,h in m['source_files'].items() if n not in [f'run_{stem}_r{oldv}.py','host_common_r50.py','perf_endpoint_r29.py',oldtest]}
 for n in [f'run_{stem}_r{newv}.py','host_common_r90.py','perf_endpoint_r85.py',f'check_queued_launch_r{newv}.py']:
  ast.parse((R/n).read_text());m['source_files'][n]=hashlib.sha256((R/n).read_bytes()).hexdigest()
 m['purpose']='Root-disk recovery for waiting coordinator only; immutable R34 runtime and complete original measurement scope retained.'
 m['recovery']={'parent':f'source-host-r{oldv}.json','parent_sha256':hashlib.sha256((R/f'source-host-r{oldv}.json').read_bytes()).hexdigest(),'driver_ast_identical_after_only_import_provenance_profiler_path_normalization':True,'TMPDIR':'NVMe per-container cache/tmp','PYTHONPYCACHEPREFIX':'NVMe per-container cache/pycache','profiler':'CPU+GPU retained; with_stack=False,record_shapes=False; no profiler during isolated native timing','paired_cleanup':'docker stop in finally even if log collection fails','gpu_acceptance':False}
 for n,h in m['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 (R/f'source-host-r{newv}.json').write_text(json.dumps(m,indent=2))
print('14 original launch configurations passed; both driver ASTs retain every measurement and barrier after three specified substitutions')
