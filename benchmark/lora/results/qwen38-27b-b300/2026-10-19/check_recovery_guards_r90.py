import ast, json, tempfile
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent
results=[]
for rev in (84,85,86,90):
 ns={};exec(compile((R/f'host_common_r{rev}.py').read_text(),'helper','exec'),ns)
 # All four helpers must attempt stop even if log reading/writing raises.
 calls=[];ns['run']=lambda c,*a:calls.append(c)
 with tempfile.TemporaryDirectory() as t:
  ns['R']=Path(t);p=Path(t)/'test-final.log';p.write_text('preserved')
  try:ns['stop'](['test'])
  except FileExistsError:pass
  else:raise AssertionError('expected injected collection failure')
  assert calls==[['docker','stop','test']]
 # Inspect actual command construction for the single-node helpers.
 if rev in (84,85,90):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);ns['R']=root;calls=[]
   (root/'launch-qwen38-tp2-d-r28.json').write_bytes((R/'launch-qwen38-tp2-d-r28.json').read_bytes())
   def tree(src,dst):dst.mkdir(parents=True)
   def file(src,dst):Path(dst).write_text('{}')
   with patch('shutil.copytree',tree),patch('shutil.copyfile',file):
    ns['launch']('check',[2],1,31200,ns['BASE'],extra=['--max-total-tokens','65536'])
   cmd=calls[0];assert 'TMPDIR=/root/.cache/tmp' in cmd and 'PYTHONPYCACHEPREFIX=/root/.cache/pycache' in cmd
   assert str(root/'caches/check')+':/root/.cache' in cmd
   assert (root/'caches/check/tmp').is_dir()
 results.append({'helper':rev,'stop_after_log_failure':True,'NVMe_launch_test':rev!=86})
for old,new in [('perf_endpoint_r29.py','perf_endpoint_r85.py'),('perf_pd_r31.py','perf_pd_r81.py'),('profile_mixed_cache_r56.py','profile_mixed_cache_r84.py')]:
 text=(R/new).read_text().replace("'record_shapes':False,'with_stack':False","'record_shapes':True")
 assert ast.dump(ast.parse(text))==ast.dump(ast.parse((R/old).read_text())),new
results.append({'profiles_AST_unchanged_except_stack_shapes':3,'natural_SSE_measurement_unchanged':True})
(R/'recovery-guards-r90.json').write_text(json.dumps({'passed':True,'scope':'CPU construction and injected cleanup failure; not GPU acceptance','checks':results},indent=2))
print(json.dumps(results))
