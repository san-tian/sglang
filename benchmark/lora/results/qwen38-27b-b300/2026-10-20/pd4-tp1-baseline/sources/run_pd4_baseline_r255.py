"""Run baseline phases serially; retain each phase failure and never overlap timings."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
src=R/'measure_pd4_baseline_r254.py'
suite={'started_unix_s':time.time(),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'phases':[],'complete':False}
(E/'measurement-source.py').write_bytes(src.read_bytes())
for phase in ['smoke','speed','natural','capacity','slots']:
 assert hashlib.sha256(src.read_bytes()).hexdigest()==suite['source_sha256']
 with (E/(phase+'.log')).open('x') as f:
  start=time.time();p=subprocess.run(['python3','-u',str(src),phase],stdout=f,stderr=subprocess.STDOUT)
 suite['phases'].append({'phase':phase,'exit_code':p.returncode,'started_unix_s':start,'ended_unix_s':time.time()})
 (E/'suite-progress.json').write_text(json.dumps(suite,indent=2))
 print(phase,p.returncode,flush=True)
 if phase=='smoke' and p.returncode:break
suite['complete']=len(suite['phases'])==5 and all(x['exit_code']==0 for x in suite['phases'])
suite['ended_unix_s']=time.time();(E/'suite-finished.json').write_text(json.dumps(suite,indent=2))
