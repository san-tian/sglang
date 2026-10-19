"""Ten-step mixed profile; extra natural requests are excluded from measured SSE matrices."""
import argparse,concurrent.futures,json,time
from pathlib import Path
from compare_merged import call
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
adapters={'real':'/work/qwen38-mol-experiment/adapters/real_r16','half':'/work/qwen38-mol-merged-bf16-20261019/adapters/real_half','gdn':'/work/qwen38-mol-experiment/adapters/gdn_fused'}
loaded=[]
try:
 for name,path in adapters.items():call(a.url,'/load_lora_adapter',{'lora_name':name,'lora_path':path});loaded.append(name)
 corpus=json.loads(Path('/work/qwen38-mol-speed-20261019/corpus-mixed.json').read_text());ids=max(corpus,key=lambda x:len(x['input_ids']))['input_ids'][:128]
 req={'input_ids':[ids]*4,'lora_path':[None,'real','half','gdn'],'sampling_params':{'temperature':0,'max_new_tokens':32,'ignore_eos':True}}
 for i in range(2):
  call(a.url,'/flush_cache',{});result=call(a.url,'/generate',req);(a.output/f'warm-{i}.json').write_text(json.dumps({'request':req,'response':result}));assert len(result)==4 and all(len(r['output_ids'])==32 for r in result)
 call(a.url,'/flush_cache',{})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  f=ex.submit(call,a.url,'/start_profile',{'output_dir':str(a.output/'profile'),'num_steps':10,'activities':['CPU','GPU'],'record_shapes':False,'with_stack':False});time.sleep(1)
  result=call(a.url,'/generate',req);(a.output/'profile-request.json').write_text(json.dumps({'request':req,'response':result}));assert len(result)==4 and all(len(r['output_ids'])==32 for r in result)
  (a.output/'profile-result.json').write_text(json.dumps(f.result()))
finally:
 for name in reversed(loaded):call(a.url,'/unload_lora_adapter',{'lora_name':name})
