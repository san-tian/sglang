"""Summarize actual GPU trace events; duration shares are not end-to-end latency shares."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
traces=sorted(a.directory.rglob('*.trace.json.gz'))+sorted(a.directory.rglob('*.trace.json'))
assert traces, f'No actual profiler trace in {a.directory}'
def category(name):
 n=name.lower()
 if '_round_merged_weight' in n:return 'LoRA scale/add/BF16 store'
 if any(s in n for s in ['nccl','allreduce','all_reduce','cross_device','reduce_scatter','all_gather']):return 'collective'
 if any(s in n for s in ['gated_delta','delta_rule','chunk_fwd','recurrent','causal_conv']):return 'GDN / convolution'
 if any(s in n for s in ['attention','flash_fwd','flash::','attn','softmax']):return 'attention / softmax'
 if 'sgemm' in n:return 'FP32 GEMM'
 if any(s in n for s in ['gemm','matmul','cutlass','gemv']):return 'GEMM / GEMV (dtype not inferred)'
 if any(s in n for s in ['copy','memcpy','index','scatter','gather']):return 'copy / routing'
 return 'other'
result=[]
for file in traces:
 opener=gzip.open if file.suffix=='.gz' else open
 with opener(file,'rt') as f:doc=json.load(f)
 kernels=[e for e in doc.get('traceEvents',[]) if e.get('cat')=='kernel' and e.get('ph')=='X' and isinstance(e.get('dur'),(int,float))]
 assert kernels, f'No actual CUDA kernels in {file}'
 names=collections.defaultdict(lambda:[0,0.]);groups=collections.defaultdict(lambda:[0,0.])
 for e in kernels:
  for bucket,key in [(names,e['name']),(groups,category(e['name']))]:
   bucket[key][0]+=1;bucket[key][1]+=e['dur']
 total=sum(e['dur'] for e in kernels)
 def rows(bucket):return [{'name':n,'count':v[0],'summed_kernel_ms':v[1]/1000,'summed_kernel_share':v[1]/total} for n,v in sorted(bucket.items(),key=lambda x:-x[1][1])]
 result.append({'file':str(file),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'gpu_kernel_count':len(kernels),'gpu_devices':sorted(set(str(e.get('args',{}).get('device')) for e in kernels)),'categories':rows(groups),'top_kernels':rows(names)[:30]})
a.output.write_text(json.dumps({'scope':'actual CUDA kernel events; summed durations may overlap, so shares are not wall-time attribution; heuristic categories retain exact names','traces':result},indent=2))
print(json.dumps({'traces':len(result),'kernel_events':sum(r['gpu_kernel_count'] for r in result),'output':str(a.output)}))
