"""One-variable colocated MTP-off control; no changes to running MTP cases."""
import json,shutil,subprocess,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'fp8-mtp-r228';name='qwen38-fp8-off-colocated-r230'
assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
used=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<2000,used
cmd=json.loads((E/'launch-qwen38-fp8-mtp-colocated-r228.json').read_text())
def change(f,v):cmd[cmd.index(f)+1]=str(v)
for f in ['--speculative-algorithm','--speculative-num-steps','--speculative-eagle-topk','--speculative-num-draft-tokens']:
 i=cmd.index(f);del cmd[i:i+2]
change('--name',name);change('--gpus','device=3');change('--port',38800);change('--nccl-port',38820);change('--served-model-name','qwen38-fp8-off-r230')
cache=E/'caches'/name
def skip(d,ns):return [x for x in ns if not (Path(d)/x).is_file() and not (Path(d)/x).is_dir() and not (Path(d)/x).is_symlink()]
shutil.copytree('/opt/dlami/nvme/qwen38-mol-development-20261019/jit-cache-template',cache,symlinks=True,ignore=skip);(cache/'tmp').mkdir(exist_ok=True)
for i,v in enumerate(cmd):
 if v.endswith(':/root/.cache'):cmd[i]=str(cache)+':/root/.cache'
(E/('launch-'+name+'.json')).write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True);subprocess.run(['docker','start',name],check=True)
(E/'started-off.json').write_text(json.dumps({'unix_s':time.time(),'gpu':3,'note':'Only MTP flags removed versus exact colocated user flags.'},indent=2))
