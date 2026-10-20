"""Focused logprob replay of the newly observed mixed-LoRA token divergence."""
import concurrent.futures as cf,importlib.util,json,math,statistics,sys,threading,time,traceback
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
for _ in range(2400):
 if (E/'residency-paced/complete.json').exists():break
 time.sleep(1)
else:raise TimeoutError('residency')
assert json.loads((E/'residency-paced/complete.json').read_text())['passed']
time.sleep(2)
sys.argv=['measure_pd4_baseline_r254.py','slots-precision-r266']
spec=importlib.util.spec_from_file_location('pd4_native_precision',R/'measure_pd4_baseline_r254.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
scaled=[{'name':f'slot-{i:02d}','path':f'/work/qwen38-mol-multigpu-20261019/fp8-mtp-r228/adapters-r239/scaled-{i:02d}'} for i in range(32)]
mon=threading.Thread(target=m.monitor);loaded=[]
def stats(a):
 a=sorted(a);return {'count':len(a),'mean_abs_nat':statistics.mean(a),'p99_abs_nat':a[math.ceil(.99*len(a))-1],'max_abs_nat':max(a)} if a else {'count':0}
try:
 m.save('server-info.json',{w['name']:m.call(w['url'],'/get_server_info') for w in m.WORKERS});mon.start()
 m.control('/unload_lora_adapter',{'lora_name':'main'})
 for s in scaled:m.control('/load_lora_adapter',{'lora_name':s['name'],'lora_path':s['path']});loaded.append(s)
 m.control('/flush_cache?timeout=60',{})
 barrier=threading.Barrier(128)
 with cf.ThreadPoolExecutor(max_workers=128) as pool:
  fs=[pool.submit(m.stream,m.prompt(128,i),32,scaled[i//4]['name'],f'mixed-lp-i{i}',i%4,(i%4+i//4)%4,barrier,True) for i in range(128)]
  mixed=[f.result() for f in fs]
 m.batch(128,32,[s for s in [None]+[x['name'] for x in scaled[:16]] for _ in range(4)],'base-plus16-replay')
 comparisons=[];prefill=[];first=[];decode=[]
 for identity in [0,15,31]:
  for d in range(4):
   k=(d-identity)%4;i=4*identity+k;m.control('/flush_cache?timeout=60',{})
   ref=m.stream(m.prompt(128,i),32,scaled[identity]['name'],f'single-lp-id{identity}-d{d}',k,d,logprobs=True);got=mixed[i]
   p=[abs(a[0]-b[0]) for a,b in zip(got['prefill_meta']['input_token_logprobs'],ref['prefill_meta']['input_token_logprobs']) if a[0] is not None and b[0] is not None]
   a,b=got['meta_info']['output_token_logprobs'],ref['meta_info']['output_token_logprobs'];assert len(a)==len(b)==32
   lcp=next((j for j,(x,y) in enumerate(zip(got['output_ids'],ref['output_ids'])) if x!=y),32)
   ds=[abs(x[0]-y[0]) for x,y in zip(a[1:lcp],b[1:lcp])];prefill.extend(p);first.append(abs(a[0][0]-b[0][0]));decode.extend(ds)
   comparisons.append({'identity':identity,'d':d,'mixed_label':got['label'],'single_label':ref['label'],'whole_output_equal':lcp==32,'common_prefix_tokens':lcp,'first_divergence_one_based':lcp+1 if lcp<32 else None,'prefill':stats(p),'first_token_abs_nat':abs(a[0][0]-b[0][0]),'decode_common_prefix':stats(ds)})
 report={'passed_execution':True,'comparisons':comparisons,'whole_output_equal':sum(c['whole_output_equal'] for c in comparisons),'total_comparisons':12,'prefill':stats(prefill),'first_token':stats(first),'decode_common_prefix':stats(decode),'scope':'Mixed32-LoRA vs single replay; 32 naturally generated tokens. Compare same-prefix same-token logprobs only, excluding first divergence. Not full teacher-forced decode or independent merged-model acceptance.'}
 ds=report['decode_common_prefix'];report['passes_old_common_prefix_gate']=ds['mean_abs_nat']<=.01 and ds['p99_abs_nat']<=.1
 m.save('comparison.json',report)
except BaseException:
 m.save('failed.json',{'traceback':traceback.format_exc(),'unix_s':time.time()});raise
finally:
 for s in loaded:
  try:m.control('/unload_lora_adapter',{'lora_name':s['name']})
  except Exception as e:m.append('cleanup-errors.jsonl',{'identity':s['name'],'error':repr(e)})
 try:m.control('/load_lora_adapter',{'lora_name':'main','lora_path':m.AD})
 finally:m.STOP.set();mon.join(30);m.PPOOL.shutdown(wait=True)
if not (m.OUT/'failed.json').exists():m.save('complete.json',{'passed':True,'requests':208,'main_restored':True,'unix_s':time.time()})
