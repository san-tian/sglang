"""CPU natural SSE rejection controls and six complete launch constructions."""
import ast,copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
from perf_million_cache_r97 import metrics
R=Path(__file__).resolve().parent
raw={'request':{'input_ids':[2,3],'stream':True,'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':3}},'http_elapsed_s':4.,'events':[{'elapsed_s':float(n),'data':{'output_ids':list(range(10,10+n))}} for n in range(1,4)]}
raw['events'][-1]['data']['meta_info']={'prompt_tokens':2,'completion_tokens':3,'finish_reason':{'type':'length','length':3},'cached_tokens':0}
assert metrics(raw,2,3)['tpot_s']==1
cases={'valid':True}
def reject(name,change):
    bad=copy.deepcopy(raw);change(bad)
    try:metrics(bad,2,3)
    except (AssertionError,KeyError,IndexError,TypeError):cases[name]=True
    else:raise AssertionError(name)
reject('forced_processor',lambda r:r['request'].update(custom_logit_processor='x'))
reject('wrong_input_length',lambda r:r['request'].update(input_ids=[2]))
reject('wrong_sampling',lambda r:r['request']['sampling_params'].update(temperature=1))
reject('rewrite_SSE_prefix',lambda r:r['events'][1]['data'].update(output_ids=[42,11]))
reject('truncated_output',lambda r:r['events'][-1]['data'].update(output_ids=[10,11]))
reject('nonfinite_time',lambda r:r['events'][1].update(elapsed_s=float('nan')))
reject('backward_time',lambda r:r['events'][1].update(elapsed_s=.5))
reject('false_HTTP_elapsed',lambda r:r.update(http_elapsed_s=2.5))
reject('early_finish',lambda r:r['events'][-1]['data']['meta_info'].update(finish_reason={'type':'stop'}))
ns={};exec(compile((R/'host_common_r90.py').read_text(),'host_common_r90.py','exec'),ns)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);ns['R']=root;(root/'launch-qwen38-tp2-d-r28.json').write_bytes((R/'launch-qwen38-tp2-d-r28.json').read_bytes())
    commands=[];ns['run']=lambda cmd,*a:commands.append(cmd)
    for i,side in enumerate(['nocache','cached','m','base','cached','nocache']):
        model=ns['MERGES']+'real_r16' if side=='m' else ns['BASE']+'/' if side=='base' else ns['BASE']
        extra=['--context-length','1000000','--max-total-tokens','1001024','--chunked-prefill-size','8192','--json-model-override-args','{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}}']
        if side=='cached':extra+=['--lora-merged-bf16-cache-size-mb','36864']
        with patch.object(ns['shutil'],'copytree',lambda src,dst:dst.mkdir(parents=True)),patch.object(ns['shutil'],'copyfile',lambda src,dst:Path(dst).write_text('{}')):
            ns['launch'](f'qwen38-million-{i}-{side}-r97',[2],1,31500,model,memory=.8,mamba=16,requests=1,extra=extra,env={'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN':'1'})
        cmd=commands[-2]
        assert 'sha256:c342c0de02976dd3fa3ca83cf1cdd5323a84ee0c82c775dfe1949ba78ae877f7' in cmd
        assert cmd[cmd.index('--gpus')+1]=='"device=2"'
        def resolved(flag):return cmd[len(cmd)-1-cmd[::-1].index(flag)+1]
        for flag,value in [('--context-length','1000000'),('--max-total-tokens','1001024'),('--chunked-prefill-size','8192'),('--max-mamba-cache-size','16'),('--max-running-requests','1'),('--mem-fraction-static','0.8')]:assert resolved(flag)==value
        assert ('--enable-lora' in cmd)==(side in ['cached','nocache'])
        assert ('--lora-merged-bf16-cache-size-mb' in cmd)==(side=='cached')
        assert all(e in cmd for e in ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1','TMPDIR=/root/.cache/tmp','PYTHONPYCACHEPREFIX=/root/.cache/pycache'])
        cases[f'launch-{i}-{side}']=True
runner=ast.parse((R/'run_million_cache_r97.py').read_text())
assert any(isinstance(n,ast.Constant) and n.value=='isolated-speed-finished-r69.json' for n in ast.walk(runner))
print(json.dumps({'passed':True,'CPU_cases':len(cases),'cases':cases,'scope':'Parser/launch preparation only; no GPU execution, million-context performance or precision acceptance.'},indent=2))
