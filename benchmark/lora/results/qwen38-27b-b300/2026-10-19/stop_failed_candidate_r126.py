"""Stop only the owned R118 speed client after the R125 numerical rejection."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

R = Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019')
name = 'qwen38-dcp8x2-m-real_r16-r118'
expected = '5a39ebffc8d48cc6e1f000522652dec7826169f9962fe726efa8ae8fc711cec4'
output = R/'candidate-numerical-rejection-stop-r126.json'
assert not output.exists()
report = R/'tp8-vs-dcp8x2-candidate-prefill-r125.json'
data = json.loads(report.read_text())
assert data['passed'] is False and data['prefill']['count'] == 49744
inspect = json.loads(subprocess.check_output(['docker','inspect',name]))[0]
assert inspect['Id'] == expected and inspect['State']['Running']
table = subprocess.check_output(['docker','top',name,'-eo','pid,ppid,args'],text=True)
clients = [row for row in table.splitlines()[1:]
           if 'python3 /work/qwen38-mol-multigpu-20261019/perf_endpoint_r85.py ' in row]
assert len(clients) == 1, table
pid = int(clients[0].split()[0])
argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
assert [x.decode() for x in argv] == [
    'python3','/work/qwen38-mol-multigpu-20261019/perf_endpoint_r85.py',
    '--url','http://127.0.0.1:30702','--output',
    '/work/qwen38-mol-multigpu-20261019/perf-dcp8x2-m-r118','--profile']
scene = {'time_unix':time.time(),'reason':'Verified R125 cross-topology numerical rejection; not a timeout',
         'failed_report':report.name,'failed_report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
         'prefill':data['prefill'],'container':name,'container_id':expected,
         'image':inspect['Image'],'state_before':inspect['State'],
         'native_client_pid':pid,'native_client_argv':[x.decode() for x in argv],
         'docker_top_before':table,'signal':'SIGTERM to owned performance client only',
         'full_goal_complete':False}
with output.open('x') as f:json.dump(scene,f,indent=2)
with (R/'candidate-numerical-rejection-engine-r126.log').open('wb') as f:
    subprocess.run(['docker','logs','--timestamps',name],stdout=f,stderr=subprocess.STDOUT,check=True)
os.kill(pid,signal.SIGTERM)
print(json.dumps(scene))
