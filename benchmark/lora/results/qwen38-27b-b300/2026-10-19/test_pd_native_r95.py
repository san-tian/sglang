"""Adversarial read-only parser replay on copies of actual two-request PD smoke."""
import argparse,json,shutil,tempfile
from pathlib import Path
from audit_pd_native_r95 import native,read,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
source=a.root/'perf-pd2diag-m-r80';names=['requests.jsonl','groups.jsonl','summary.json','server-info.json','prefill-server-info.json']
labels=[json.loads(s)['label'] for s in (source/'requests.jsonl').read_text().splitlines()];names +=[s+'.json' for s in labels]
before={n:sha(source/n) for n in names};results=[]
def change_json(folder,name,fn):
 p=folder/name;data=read(p);fn(data);p.write_text(json.dumps(data))
def change_lines(folder,name,fn):
 p=folder/name;data=[json.loads(s) for s in p.read_text().splitlines()];fn(data);p.write_text(''.join(json.dumps(v)+'\n' for v in data))
cases=[('valid',None),
 ('wrong_bootstrap_room',lambda f:change_json(f,labels[0]+'.json',lambda d:d['prefill']['request'].update(bootstrap_room=1))),
 ('forced_native_output',lambda f:change_json(f,labels[0]+'.json',lambda d:[d[k].update(custom_logit_processor='forced') for k in ['decode_request']]+[d['prefill']['request'].update(custom_logit_processor='forced')])),
 ('wrong_SSE_time',lambda f:change_json(f,labels[0]+'.json',lambda d:d['decode_events'][0].update(elapsed_s=.00001))),
 ('truncated_decode',lambda f:change_json(f,labels[0]+'.json',lambda d:d['decode_events'][-1]['data']['output_ids'].pop())),
 ('false_throughput',lambda f:change_json(f,'summary.json',lambda d:d['points'][0].update(throughput_tokens_s_mean=1e9))),
 ('false_P95',lambda f:change_json(f,'summary.json',lambda d:d['points'][0]['e2e_s'].update(p95=0))),
 ('duplicate_groups',lambda f:change_lines(f,'groups.jsonl',lambda d:d.__setitem__(1,d[0]))),
 ('nonfinite_TTFT',lambda f:change_lines(f,'requests.jsonl',lambda d:d[0].update(ttft_s=float('nan')))),
 ('mismatched_prefill_token',lambda f:change_json(f,labels[0]+'.json',lambda d:d['prefill']['response']['output_ids'].__setitem__(0,-1)))]
for name,mutate in cases:
 with tempfile.TemporaryDirectory() as temp:
  directory=Path(temp)
  for n in names:shutil.copyfile(source/n,directory/n)
  if mutate:mutate(directory)
  failed=False
  try:native(directory,'m',2,True)
  except (AssertionError,KeyError,IndexError,TypeError):failed=True
  assert failed==(mutate is not None),name
  results.append({'case':name,'expected_rejection':mutate is not None,'passed':True})
assert before=={n:sha(source/n) for n in names}
report={'passed':True,'cases':results,'actual_input_files_sha256':before,'source_sha256':sha(Path(__file__)),'scope':'CPU-only parser replay on copies of existing2P2D M smoke, not new GPU/PD4 accuracy or speed acceptance.'}
(a.root/'pd-native-parser-tests-r95.json').write_text(json.dumps(report,indent=2));print('Passed',len(results),'parser positive/negative cases; original smoke files unchanged')
