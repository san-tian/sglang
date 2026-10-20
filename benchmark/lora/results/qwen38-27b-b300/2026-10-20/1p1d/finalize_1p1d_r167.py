"""CPU-only final audits triggered by actual completed experiment markers."""
import json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
def wait(name,failures):
 while not (R/name).exists():
  for item in failures:
   if (R/item).exists():raise RuntimeError((R/item).read_text())
  time.sleep(3)
def run(script,log):
 with (R/log).open('x') as f:subprocess.run(['python3',str(R/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 wait('1p1d-control-finished-r161.json',['measure-1p1d-r160/failed.json','1p1d-control-failed-r161.json'])
 run('audit_1p1d_complete_r166.py','1p1d-main-audit-r166.log')
 wait('1p1d-slot-sweep-finished-r163.json',['1p1d-slot-sweep-failed-r163.json'])
 run('audit_1p1d_slots_r164.py','1p1d-slot-sweep-audit-r164.log')
 for name in ['1p1d-main-audit-r166.json','1p1d-slot-sweep-audit-r164.json']:
  assert json.loads((R/name).read_text())['passed']
 with (R/'1p1d-audits-finished-r167.json').open('x') as f:json.dump({'passed':True,'unix_s':time.time(),'full_goal_complete':False},f)
except BaseException:
 with (R/'1p1d-audits-failed-r167.json').open('x') as f:json.dump({'traceback':traceback.format_exc(),'unix_s':time.time()},f,indent=2)
 raise
