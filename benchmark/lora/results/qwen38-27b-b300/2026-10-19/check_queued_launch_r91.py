"""Construct each R69 launch without Docker, CUDA or model loading."""
import ast,json,tempfile
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent
IMAGE='sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7'
runner=ast.parse((R/'run_isolated_speed_r91.py').read_text())
launches=[n for n in ast.walk(runner) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='launch']
assert len(launches)==1 and any(k.arg=='memory' and isinstance(k.value,ast.Constant) and k.value.value==.52 for k in launches[0].keywords)
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);ns={};exec(compile((R/'host_common_r90.py').read_text(),'host_common_r90.py','exec'),ns)
 ns['R']=root
 (root/'launch-qwen38-tp2-d-r28.json').write_text((R/'launch-qwen38-tp2-d-r28.json').read_text())
 commands=[];ns['run']=lambda cmd,*args:commands.append(cmd)
 def copytree(src,dst):dst.mkdir(parents=True)
 def copyfile(src,dst):Path(dst).write_text('{}')
 for position,side in enumerate(['nocache','cached','m','base','cached','nocache']):
  model=ns['MERGES']+'real_r16' if side=='m' else ns['BASE']+'/' if side=='base' else ns['BASE']
  extra=['--max-total-tokens','65536']+(['--lora-merged-bf16-cache-size-mb','36864'] if side=='cached' else [])
  with patch.object(ns['shutil'],'copytree',copytree),patch.object(ns['shutil'],'copyfile',copyfile):
   ns['launch'](f'qwen38-isolated-{position}-{side}-r69',[2],1,31200,model,extra=extra,memory=.52)
  cmd=commands[-2]
  assert IMAGE in cmd and cmd[cmd.index('--gpus')+1]=='"device=2"'
  for flag,value in [('--mem-fraction-static','0.52'),('--max-total-tokens','65536'),('--max-mamba-cache-size','256'),('--tp-size','1')]:assert cmd[cmd.index(flag)+1]==value
  assert ('--enable-lora' in cmd)==(side in ['cached','nocache'])
  assert ('--lora-merged-bf16-cache-size-mb' in cmd)==(side=='cached')
  print(json.dumps({'position':position,'side':side,'image':IMAGE,'memory_fraction':.52,'KV_requested':65536,'SSM_slots':256,'passed':True}))
print('Six CPU launch construction cases passed; no GPU execution or acceptance.')
