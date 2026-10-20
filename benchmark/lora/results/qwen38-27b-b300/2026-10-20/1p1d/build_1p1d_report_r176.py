"""Summarize final audited data; no timings are invented or filtered."""
import csv,datetime,hashlib,json,statistics,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
DEST=Path('/home/dev/Deployment/qwen38-mol-sglang/benchmark/lora/results/qwen38-27b-b300/2026-10-20/1p1d')
def j(n):return json.loads((DEST/n).read_text())
def member(folder,name):
 with tarfile.open(DEST/(folder+'.tgz')) as t:return t.extractfile(folder+'/'+name).read().decode()
def lines(folder,name):return [json.loads(s) for s in member(folder,name).splitlines()]
def q(xs,k):
 xs=sorted(xs);p=(len(xs)-1)*k;i=int(p);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(p-i)
kv=j('1p1d-KV-observed-audit-r185.json');assert kv['passed']
live=j('final-live-state-r175.json');basics=j('model-basics-r187.json')
a=j('1p1d-main-audit-r171.json');b=j('1p1d-slot-sweep-audit-r172.json');assert a['passed'] and b['passed']
points=[]
for side,folder in [('D','measure-1p1d-r160'),('M','measure-1p1d-merged-r168')]:
 groups=lines(folder,'groups.jsonl');reqs=lines(folder,'requests.jsonl')
 for nin in [128,2048]:
  for c in [1,32,64]:
   gs=[g for g in groups if g['label'].startswith(f'speed-in{nin}-c{c}-r')];assert len(gs)==3
   rr=[r for r in reqs if any(r['label'].startswith(g['label']+'-i') for g in gs)];assert len(rr)==3*c
   points.append({'control':side,'input_tokens':nin,'output_tokens':128 if nin==128 else 256,'concurrency':c,'rounds':3,'requests':len(rr),'output_tokens_s_mean':statistics.mean(g['output_tokens_s'] for g in gs),'output_tokens_s_rounds':[g['output_tokens_s'] for g in gs],'ttft_p95_s':q([r['ttft_s'] for r in rr],.95),'tpot_p95_ms':1000*q([r['tpot_s'] for r in rr],.95),'e2e_p95_s':q([r['e2e_s'] for r in rr],.95)})
with (DEST/'throughput-1p1d.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(points[0]),lineterminator="\n");w.writeheader();w.writerows(points)
slots=[]
for gpu,cpu,folder in [(16,32,'measure-1p1d-r160'),(32,64,'measure-1p1d-slots32-r169')]:
 groups=lines(folder,'groups.jsonl')
 for g in groups:
  if not g['label'].startswith('slots-'):continue
  slots.append({'GPU_slots_per_role':gpu,'registered_adapters_per_role':cpu,**{k:g[k] for k in ['label','input_tokens','output_tokens','concurrency','distinct_adapters','elapsed_s','output_tokens_s']}})
with (DEST/'lora-slots-1p1d.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(slots[0]),lineterminator="\n");w.writeheader();w.writerows(slots)
summary={'as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'topology':'P GPU0 TP1 + D GPU1 TP1, two B300 GPUs','primary_GPU_slots_per_role':16,'primary_registered_adapter_limit_per_role':32,'validated_alternative_GPU_slots_per_role':32,'validated_alternative_registered_limit_per_role':64,'native_context_limit_configured':262144,'native_performance_points':points,'main_audit':'1p1d-main-audit-r171.json','slot_audit':'1p1d-slot-sweep-audit-r172.json','shared_request_KV_pool_tokens_bound':a['shared_request_KV_pool_tokens_bound'],'current_restored_KV_per_role':{role:v['max_total_num_tokens'] for role,v in live['server_info'].items()},'model_basics':basics,'official_model_reference':j('official-model-reference-r187.json'),'allocated_capacity_per_role':a['allocated_capacity_per_role'],'alternative_allocated_KV_per_role':b['allocated_KV_per_role'],'teacher_forced_precision':a['teacher_forced_precision'],'natural_D_M_requests_equal':a['formal_natural_D_M_requests_equal'],'near_pool_resident_test':kv,'full_goal_complete':False}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=['# 两卡 1P1D 实测\n','主方案：P GPU0/TP1＋D GPU1/TP1；原始 base 动态挂载真实 rank16 LoRA。实验机独占两张 B300，其余六卡空闲。当前保持16 GPU槽位／最多登记32 adapters，两端各加载main；32 GPU槽位／64登记是另外完成的工作点。两端槽位与KV容量均不能相加。\n',
'官方Qwen3.8-27B模型卡：原生上下文262144，扩展上限1000000（https://huggingface.co/Qwen/Qwen3.8-27B）。本地config与官方逐字节一致；沿用qwen3_5架构字段。官方标称27B，本地完整checkpoint为27.781B个BF16元素（含视觉/MTP），18个权重分片共55.56GB/51.75GiB；KV/SSM/LoRA缓存另计。见model-basics-r187.json与official-model-reference-r187.json。\n',
'模型目录名 Qwen3.8-27B，实际config为Qwen3_5ForConditionalGeneration/qwen3_5_text，dense，非MoE。本轮为BF16 deterministic eager、merged_bf16、triton attention、torch GEMM、36GiB/role active投影缓存、关闭CUDA graph、NIXL/UCX、mem-fraction .80。两容器均可见GPU0/1，角色分别用base-gpu-id 0/1，避免CUDA枚举不一致。\n',
'## 速度\n','输出token/s，投影缓存预热、每批flush KV，1次预热＋3轮正式测试。单adapter greedy生成，ignore_eos=true固定长度，不强制token内容；D/M分别运行、同两卡与相同KV token配额。包含PD传输、排队及原始响应采集开销。\n',
'| 输入/输出 | 并发 | D tok/s | M tok/s | D/M | D TTFT P95(s) | D TPOT P95(ms) |\n','|---|---:|---:|---:|---:|---:|---:|\n']
for p in a['dynamic_vs_merge']:
 rows.append(f"| {p['input_tokens']}/{p['output_tokens']} | {p['concurrency']} | {p['output_tokens_s_mean']:.2f} | {p['merged_output_tokens_s_mean']:.2f} | {100*p['dynamic_fraction_of_merged']:.2f}% | {p['ttft_p95_s']:.3f} | {p['tpot_p95_ms']:.2f} |\n")
rows+=['\n短输入c64动态正式三轮为687.67/1034.95/1034.38，均值919.00；存在轮次波动，全部保留。历史TP1的KV配额更小，不能将跨组差值归因于PD。没有完成相同两卡双TP1副本对照，不能宣称1P1D为全局最优。详见[吞吐与P95 CSV](throughput-1p1d.csv)。\n',
'## KV与LoRA槽位\n','| GPU槽位/role | 登记上限/role | P分配KV tokens | D分配KV tokens |\n','|---:|---:|---:|---:|\n']
rows.append(f"|16|32|{a['allocated_capacity_per_role']['prefill']['actual_allocated_KV_tokens']}|{a['allocated_capacity_per_role']['decode']['actual_allocated_KV_tokens']}|\n")
rows.append(f"|32|64|{b['allocated_KV_per_role']['prefill']['allocated_KV_tokens']}|{b['allocated_KV_per_role']['decode']['allocated_KV_tokens']}|\n")
rows.append(f"\n恢复后的当前实际分配：P {live['server_info']['prefill']['max_total_num_tokens']:,} / D {live['server_info']['decode']['max_total_num_tokens']:,} tokens；重启时自动分配有小幅浮动。主测速D/M使用的是表中初始配额。\n")
rows.append(f"\n接近池容量的补测：32×(32768输入＋2048输出)全部完成，总prompt 1048576 tokens。最多观察到{int(kv['max_running_requests_observed'])}路同时运行；同次KV峰值{int(kv['same_scrape_peak']['metrics']['kv_used_tokens'])} tokens、运行序列总长{int(kv['same_scrape_peak']['metrics']['decode_sum_seq_lens'])} tokens。32路整批同时驻留采样数为{kv['full_prompt_resident_samples']}，原32路检测项未通过，失败记录保留。批次耗时{kv['batch_elapsed_s']:.2f}秒。[原始响应及观察值复核](1p1d-KV-observed-audit-r185.json)通过不代表原检测项通过。KV used可能含队列/缓存分配，运行序列总长单独列出；这是多请求合计，非单请求百万上下文。\n")
rows+=['\n配对池受较小端限制，分配值不是满池压力验收。主方案32×(8192输入＋512输出)全部完成；同次decode采样峰值272903 KV tokens、32运行请求，22个采样观察到整批prompt驻留。32K输入＋512输出另已完成1路。本轮context配置262144（输入＋输出），尚未在此拓扑验收满上下文质量。Mamba/SSM另占内存：conv约0.71GiB、SSM约36.14GiB/role；不计入KV tokens。\n',
'槽位池按max rank32预留，实测adapter为rank16。32/64个容量adapter是同一真实A/B权重的不同alpha版本，支持性验证不等于32/64个独立训练模型的质量验收。base混流占一个批身份；16槽位一批支持15 LoRA＋base，32槽位支持31 LoRA＋base。更多身份请求可排队/换入。两端分别成功登记32/64，且混流超过池容量及base混流均实际完成。\n',
'| GPU槽位/登记 | 混流LoRA身份 | 输入/输出 | 输出tok/s，两轮均值 |\n','|---|---:|---|---:|\n']
for p in a['slots']:rows.append(f"|16/32|{p['requested_adapter_identities']}|128/32|{p['output_tokens_s_mean']:.2f}|\n")
for p in b['speed_points']:rows.append(f"|32/64|{p['distinct_requested_adapters']}|128/32|{p['output_tokens_s_mean']:.2f}|\n")
rows+=['\n多LoRA吞吐明显下降；该组输出长度仅32，与主测速矩阵不同，包含可能的缓存填充/换入成本。源码run_merged_linear按adapter逐组调用F.linear，缓存miss时重新merge；计算路径已确认，但逐组GEMM、缓存换入和merge各自的耗时占比尚未profile。增加槽位不等于吞吐增加。详见[逐轮槽位CSV](lora-slots-1p1d.csv)。\n',
'## 精度与证据\n',f"动态D vs独立离线merge M：{a['formal_natural_D_M_requests_equal']}/582个正式自然输出序列一致。短样本teacher forcing：8例×2轮×cold/hot＝32对，32输出token；原始强制之前的prefill/首token/decode评分分别如下。\n"]
for phase,p in a['teacher_forced_precision']['scores'].items():rows.append(f"- {phase}: {p['count']}评分，mean {p['mean_abs_nat']:.8g} /P99 {p['p99_abs_nat']:.8g} /max {p['max_abs_nat']:.8g} nat。\n")
rows+=['\n输入长度17–29，仅为本拓扑回归验证，不能替代完整长文数值或任务质量验收。阈值沿用mean≤0.01/P99≤0.1 nat。动态与原始base权重不同；本轮未给base任务准确率增益。\n',
'[主原始审计](1p1d-main-audit-r171.json)、[32槽位审计](1p1d-slot-sweep-audit-r172.json)、[最终健康及配置](final-live-state-r175.json)、[全部原始文件SHA256](raw-files-sha256.json)与各组tgz保留每个自然请求P响应/D SSE、时间戳、分配日志、驻留指标及数值capture。R161测试脚本复制临时socket失败，未启动merge模型；失败证据保留，R168修复缓存复制后重新完整执行，无删改原始测速。\n',
'全局多卡优化目标未完成；旧4P4D/DP8工作按用户优先级暂缓。当前结果支持暂用1P1D，后续重点为多LoRA混流性能与正式负载/SLO验证。\n']
(DEST/'README.md').write_text('\n'.join(rows).replace('|\n\n|','|\n|'))
manifest=j('manifest.json');manifest['as_of_utc']=summary['as_of_utc']
manifest['files_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,sha in manifest['files_sha256'].items():assert hashlib.sha256((DEST/name).read_bytes()).hexdigest()==sha,name
print('Verified',len(manifest['files_sha256']),'artifact hashes; main points',len(points),'slot rounds',len(slots))
