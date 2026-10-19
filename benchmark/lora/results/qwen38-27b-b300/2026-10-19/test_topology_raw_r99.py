"""CPU rejection tests for original sampling-rank capture reconstruction."""
import copy,json,tempfile
from pathlib import Path
from audit_topology_original_r99 import capture_scores
case={'input_ids':[1,2]};cont=[3,4,5]
response={'output_ids':cont,'meta_info':{'prompt_tokens':2,'completion_tokens':3,'finish_reason':{'type':'length','length':3},'input_token_logprobs':[[None,1],[-.2,2]],'input_top_logprobs':[[],[[-.2,2]]]}}
rows=[{'step':n,'token':t,'logprob':-.2,'argmax':t} for n,t in enumerate(cont)]
cases={}
def run(name,tp,dp,ranks,change=None,reject=False):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);res=copy.deepcopy(response)
        for rank in ranks:
            path=root/f'x.jsonl.rank-{rank}';path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        if change:change(root,res)
        try:capture_scores(root,'x',res,case,cont,tp,dp)
        except (AssertionError,KeyError,IndexError,TypeError):assert reject,name
        else:assert not reject,name
        cases[name]=True
run('valid_tp2',2,1,[0,1])
run('missing_tp_replica',2,1,[0],reject=True)
run('wrong_rank',2,1,[0,2],reject=True)
def mutate(root,res,kind):
    p=root/'x.jsonl.rank-0';data=copy.deepcopy(rows)
    if kind=='difference':data[1]['logprob']=-.3
    if kind=='step':data[1]['step']=2
    if kind=='nan':data[0]['logprob']=float('nan')
    p.write_text(''.join(json.dumps(v)+'\n' for v in data))
run('different_rank_scores',2,1,[0,1],lambda p,r:mutate(p,r,'difference'),True)
run('wrong_step',2,1,[0,1],lambda p,r:mutate(p,r,'step'),True)
run('nonfinite_capture',2,1,[0,1],lambda p,r:mutate(p,r,'nan'),True)
run('wrong_output_path',2,1,[0,1],lambda p,r:r.update(output_ids=[3,7,5]),True)
run('nonfinite_prefill',2,1,[0,1],lambda p,r:r['meta_info']['input_token_logprobs'][1].__setitem__(0,float('nan')),True)
run('attention_dp8_one_local_sampling_rank',8,8,[5])
run('attention_dp8_extra_rank',8,8,[0,5],reject=True)
run('dcp2_sampling_replicas_contract',8,1,list(range(8)))
run('dcp2_missing_sampling_replica',8,1,list(range(7)),reject=True)
print(json.dumps({'passed':True,'CPU_cases':len(cases),'cases':cases,'scope':'Sampling-rank parser contract tests only. DCP8GPU runtime correctness remains untested.'},indent=2))
