"""CPU-only construction check: same8 GPU PD protocol, adding transport introspection."""
import ast,json,tempfile
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent
helper=ast.parse((R/'host_common_r78.py').read_text());launch=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='launch')
driver=ast.parse((R/'run_pd4_suite_r78.py').read_text());pair=next(n for n in driver.body if isinstance(n,ast.FunctionDef) and n.name=='pair')
with tempfile.TemporaryDirectory() as tmp:
 r=Path(tmp);(r/'launch-qwen38-tp2-d-r28.json').write_bytes((R/'launch-qwen38-tp2-d-r28.json').read_bytes())
 ns={'R':r,'json':json,'shutil':SimpleNamespace(copytree=lambda *a:None,copyfile=lambda *a:None),'CLIENT':'/work/qwen38-mol-multigpu-20261019/client-r33','BASE':'/work/models/Qwen3.8-27B','run':lambda *a:None,'portp':30800,'portd':30801,'bootstrap':30810,'env':{'SGLANG_DISAGGREGATION_NIXL_BACKEND':'UCX','UCX_TLS':'cuda_copy,cuda_ipc,posix,sysv,tcp','UCX_LOG_LEVEL':'info'}}
 exec(compile(ast.Module(body=[launch,pair],type_ignores=[]),'<pd4 launch>','exec'),ns)
 count=0
 for label,model in [('d',ns['BASE']),('m-real','/work/merged/real'),('m-gdn','/work/merged/gdn'),('base',ns['BASE']+'/')]:
  names=ns['pair'](label,model);assert len(names)==2
  for index,name in enumerate(names):
   c=json.loads((r/f'launch-{name}.json').read_text());assert c[c.index('--gpus')+1]=='"device=0,1,2,3,4,5,6,7"'
   assert c[c.index('--base-gpu-id')+1]==str(index*4) and c[c.index('--pid')+1]=='host'
   assert 'sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288' in c
   assert c[c.index('--tp-size')+1]=='4' and c[c.index('--disaggregation-mode')+1]==('prefill' if index==0 else 'decode')
   assert c[c.index('--disaggregation-transfer-backend')+1]=='nixl'
   assert '--disaggregation-enable-kv-checksum' in c and c[c.index('--mamba-radix-cache-strategy')+1]=='extra_buffer'
   assert all(k+'='+v in c for k,v in ns['env'].items())
   assert ('--enable-lora' in c)==(label=='d');count+=1
 print(json.dumps({'cpu_launch_cases':count,'docker_invocations':0,'gpu_acceptance':False}))
