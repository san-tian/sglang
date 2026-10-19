"""Command-construction checks only; no Docker/GPU calls or model acceptance."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
class LaunchChecks(unittest.TestCase):
 def test_all_four_controls_keep_layout_kv_and_image(self):
  namespace={};exec(compile((ROOT/'host_common_r90.py').read_text(),str(ROOT/'host_common_r90.py'),'exec'),namespace)
  image='sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7'
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);namespace['R']=root;commands=[];namespace['run']=lambda cmd,*args:commands.append(cmd)
   (root/'launch-qwen38-tp2-d-r28.json').write_bytes((ROOT/'launch-qwen38-tp2-d-r28.json').read_bytes())
   with patch('shutil.copytree'),patch('shutil.copyfile'):
    for dpa in [False,True]:
     for side in ['m','d','nocache','base']:
      model=namespace['MERGES']+'real_r16' if side=='m' else namespace['BASE']+'/' if side=='base' else namespace['BASE']
      extra=['--max-total-tokens','65536']+(['--lora-merged-bf16-cache-size-mb','12288' if dpa else '10240'] if side=='d' else [])
      namespace['launch'](f'check-{dpa}-{side}',[0,1,2,3],4,31300,model,dpa=dpa,extra=extra)
      cmd=commands[-2]
      self.assertEqual(cmd[cmd.index('--gpus')+1],'"device=0,1,2,3"')
      self.assertIn(image,cmd)
      self.assertEqual(cmd[cmd.index('--tp-size')+1],'4')
      self.assertEqual(cmd[cmd.index('--max-total-tokens')+1],'65536')
      self.assertEqual(cmd[cmd.index('--max-mamba-cache-size')+1],'256')
      self.assertEqual('--enable-lora' in cmd,side in ['d','nocache'])
      self.assertEqual('--lora-merged-bf16-cache-size-mb' in cmd,side=='d')
      self.assertEqual('--attn-dp-size' in cmd,dpa)
      if dpa:self.assertEqual(cmd[cmd.index('--attn-dp-size')+1],'4')
      self.assertEqual(commands[-1],['docker','start',f'check-{dpa}-{side}'])
if __name__=='__main__':unittest.main()
