"""CPU adversarial tests of full-length TP4-per-role original-score parsing."""
import copy,json,tempfile,unittest
from pathlib import Path
from audit_pd_precision_r94 import request_scores

class ParserChecks(unittest.TestCase):
 def exercise(self,mutation=None):
  with tempfile.TemporaryDirectory() as t:
   folder=Path(t)/'pd4-r33-full-dynamic-r30';(folder/'captures').mkdir(parents=True)
   cont=list(range(256));case={'input_ids':[2,3]};name='g0-r0-cold'
   params={'temperature':0,'max_new_tokens':256,'ignore_eos':True,'custom_params':{'continuation':cont,'capture_file':str(folder/'captures'/f'{name}-i0.jsonl.prefill'),'capture_start_step':0}}
   pr={'input_ids':case['input_ids'],'return_logprob':True,'logprob_start_len':0,'top_logprobs_num':1,'custom_logit_processor':'fixed-processor','sampling_params':params,'bootstrap_host':'127.0.0.1','bootstrap_port':30810,'bootstrap_room':23,'lora_path':'frozen-A'}
   dr=copy.deepcopy(pr);dr['sampling_params']['custom_params'].update(capture_start_step=1,capture_file=str(folder/'captures'/f'{name}-i0.jsonl.decode'))
   pm={'prompt_tokens':2,'completion_tokens':1,'input_token_logprobs':[[None,2],[-.2,3]],'input_top_logprobs':[None,[[-.2,3]]]}
   dm={'prompt_tokens':2,'completion_tokens':256,'finish_reason':{'type':'length','length':256}}
   pres={'output_ids':[0],'meta_info':pm};dres={'output_ids':cont,'meta_info':dm}
   combined=copy.deepcopy(dres);combined['meta_info'].update({k:v for k,v in pm.items() if k.startswith('input_')})
   for role,steps in [('prefill',[0]),('decode',range(1,256))]:
    for rank in range(4):
     rows=[{'step':i,'token':cont[i],'logprob':-.3,'argmax':cont[i]} for i in steps]
     (folder/'captures'/f'{name}-i0.jsonl.{role}.rank-{rank}').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   index={f'{name}-i0.jsonl.prefill':(pr,dr,pres,dres)}
   if mutation:mutation(folder,pr,dr,pres,dres,combined)
   out=request_scores(folder,index,name,0,case,cont,combined,4,True)
   self.assertEqual([len(out[p]) for p in ('prefill','prefill_last','decode')],[1,1,255])
 def rejects(self,f):
  with self.assertRaises(AssertionError):self.exercise(f)
 def test_full_256_tokens_four_ranks(self):self.exercise()
 def test_missing_rank(self):self.rejects(lambda p,*_: (p/'captures/g0-r0-cold-i0.jsonl.decode.rank-3').unlink())
 def test_wrong_capture_offset(self):self.rejects(lambda p,pr,dr,*_:dr['sampling_params']['custom_params'].update(capture_start_step=2))
 def test_room_mismatch(self):self.rejects(lambda p,pr,dr,*_:dr.update(bootstrap_room=99))
 def test_nonfinite_raw_prefill(self):self.rejects(lambda p,pr,dr,pres,*_:pres['meta_info']['input_token_logprobs'].__setitem__(1,[float('nan'),3]))
 def test_wrong_merged_response(self):self.rejects(lambda p,pr,dr,pres,dres,c:c['meta_info'].update(input_fake='changed'))
 def test_rank_disagreement(self):
  def change(p,*_):
   f=p/'captures/g0-r0-cold-i0.jsonl.decode.rank-3';rows=[json.loads(s) for s in f.read_text().splitlines()];rows[0]['argmax']=999;f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
  self.rejects(change)
 def test_wrong_natural_path(self):self.rejects(lambda p,pr,dr,pres,dres,*_:dres['output_ids'].__setitem__(2,999))
if __name__=='__main__':unittest.main()
