"""No GPU operations: verify real launch constructors and unchanged measurement AST."""
import ast,json,shutil,tempfile
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent
old=ast.parse((R/'host_common_r85.py').read_text())
new=ast.parse((R/'host_common_r118.py').read_text())
restored=(R/'host_common_r118.py').read_text().replace('source-host-r118.json','source-host-r85.json').replace('-r118','-r30').replace('sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878','sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288')
restored=ast.parse(restored)
fns=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
a,b=fns(old),fns(restored)
assert set(b)-set(a)=={'placement'}
assert all(a[n]==b[n] for n in a),[n for n in a if a[n]!=b[n]]
ns={};exec(compile(new,'host_common_r118.py','exec'),ns)
commands=[]
with tempfile.TemporaryDirectory() as td:
 tmp=Path(td)
 shutil.copyfile(R/'launch-qwen38-tp2-d-r28.json',tmp/'launch-qwen38-tp2-d-r28.json')
 ns['R']=tmp;ns['run']=lambda cmd,*args,**kwargs:commands.append(cmd)
 def tree(src,dst):Path(dst).mkdir(parents=True)
 def copy(src,dst):Path(dst).write_text('{}')
 with patch.object(shutil,'copytree',side_effect=tree),patch.object(shutil,'copyfile',side_effect=copy):
  for top,dpa in [('dcp8x2',False),('dpa8',True)]:
   for role in ['m','gdn','d','base','capacity']:
    model=ns['BASE'] if role in ['d','capacity'] else ns['BASE']+'/' if role=='base' else ns['MERGES']+('gdn_fused' if role=='gdn' else 'real_r16')
    extra=[] if dpa else ['--dcp-size','2','--dcp-comm-backend','a2a']
    if role=='capacity':extra+=['--page-size','1','--context-length','16384','--max-total-tokens',str(8191 if dpa else 32767),'--enable-metrics','--decode-log-interval','1']
    ns['launch'](f'qwen38-{top}-{role}-r118',list(range(8)),8,30704 if dpa else 30702,model,dpa=dpa,extra=extra,mamba=32 if dpa else 256)
    cmd=commands[-2]
    assert cmd[:2]==['docker','create']
    assert cmd[cmd.index('--gpus')+1]=='"device=0,1,2,3,4,5,6,7"'
    assert cmd[cmd.index('--tp-size')+1]=='8'
    assert ns['CLIENT'].endswith('/client-r30')
    assert 'sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878' in cmd
    assert ('--enable-lora' in cmd)==(role in ['d','capacity'])
    assert ('--attn-dp-size' in cmd)==dpa
    assert ('--dcp-size' in cmd)==(not dpa)
    assert '--chunked-prefill-size' in cmd and cmd[cmd.index('--chunked-prefill-size')+1]=='2048'
    assert 'TMPDIR=/root/.cache/tmp' in cmd
    assert 'PYTHONPYCACHEPREFIX=/root/.cache/pycache' in cmd
    assert cmd[cmd.index('--max-mamba-cache-size')+1]==str(32 if dpa else 256)
    if role=='capacity':
     assert cmd[-cmd[::-1].index('--max-total-tokens')]=='8191' if dpa else cmd[-cmd[::-1].index('--max-total-tokens')]=='32767'
print('10 topology/role launch configurations passed; all original helper functions AST identical after image/provenance/namespace normalization; actual placement proof is the only new helper.')
