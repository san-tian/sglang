"""CPU regression for disabled LoRA null/false without accepting invalid roles."""
import ast,hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
source=r/'audit_active_cache_recheck_r105.py';tree=ast.parse(source.read_text())
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='validate_lora_role')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'role-regression','exec'),ns)
validate=ns['validate_lora_role']
paths={'m':'/work/qwen38-mol-development-20261019/merged/real_r16',
       'base':'/work/models/Qwen3.8-27B/','d':'/work/models/Qwen3.8-27B'}
passed=0
for side,values in [('m',[None,False]),('base',[None,False]),('d',[True])]:
    for value in values:
        validate({'enable_lora':value,'model_path':paths[side]},side);passed+=1
bad=[('m',{'enable_lora':True,'model_path':paths['m']}),
     ('d',{'enable_lora':None,'model_path':paths['d']}),
     ('d',{'enable_lora':False,'model_path':paths['d']}),
     ('base',{'enable_lora':'false','model_path':paths['base']}),
     ('base',{'enable_lora':0,'model_path':paths['base']}),
     ('m',{'model_path':paths['m']}),
     ('m',{'enable_lora':None,'model_path':paths['d']})]
for side,config in bad:
    try:validate(config,side)
    except (AssertionError,KeyError):passed+=1
    else:raise AssertionError(('invalid role accepted',side,config))
report={'passed':True,'cpu_checks':passed,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'CPU audit-role serialization regression only; no GPU acceptance.'}
(r/'audit-role-normalization-r105.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
