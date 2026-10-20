"""Build the baseline report from exported, audited numbers."""
import json,statistics,sys,tarfile
from pathlib import Path
D=Path(sys.argv[1]);a=json.loads((D/'audit-r256.json').read_text());cap=json.loads((D/'residency-audit-r265.json').read_text());lp=json.loads((D/'mixed-lora-logprob-comparison.json').read_text())
speed=a['phases']['speed']['speed'];natural=a['phases']['natural']['speed'];slots=a['phases']['slots'];kv=a['phases']['smoke']['kv_tokens_per_worker']
L=['# 八卡 4P＋4D、TP1 基线','',
'已按确认切换为8张B300：GPU0–3为4个prefill实例，GPU4–7为4个decode实例，每个实例TP=1。性能与容量数据已封存；精度分歧单独记录，不能把启动/吞吐通过等同于FP8模型精度验收通过。','',
'## 固定配置','',
'|项目|配置|','|---|---|',
'|模型|Qwen/Qwen3.8-27B-FP8；revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`|',
'|硬件|8×NVIDIA B300 SXM6；每卡275,040 MiB；GPU之间均NV18；192逻辑CPU、2个NUMA节点|',
'|运行时|驱动595.91.07；PyTorch2.14.1+cu130；CUDA13.0；NIXL1.5.0；FlashInfer0.7.2；每卡功耗上限1100W|',
'|权重与缓存|FP8权重、`fp8_e4m3` KV、SSM bfloat16|',
'|MTP|EAGLE，num-steps=3、topk=1、num-draft-tokens=4|',
'|上下文|配置262,144；MTP验证预留4；输入＋输出实测上限262,140；未验收百万扩展|',
'|调度|mem-fraction-static=0.80；chunked-prefill=8192；max-running=64/实例；D端合计配置256|',
'|Mamba|full-memory-ratio=3.74；extra_buffer；PD不支持extra_buffer_lazy|',
'|LoRA|max-rank=32；真实adapter rank16；triton backend；每实例GPU批身份16、CPU缓存32|',
'|解析器|reasoning=qwen3；tool-call=qwen3_coder|',
'|内核与图|FP8、TRTLLM MHA、FlashInfer GDN默认路径；CUDA graphs开启；不使用旧BF16 deterministic eager设置|',
'|传输与分发|NIXL/UCX；原生客户端平衡选择P和D；覆盖全部16种配对；未改生产入口|','',
'实际引擎是镜像 `sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934` 加 fork `0116d3a12bc893ae43d8dbfd594280eccdc8467c` 的26个Python覆盖文件；8个实例共208项哈希重新核对。容器继承的.git/OCI标签仍为上游，不能当作实际覆盖代码版本。','',
'## 三轮性能','',
'每组1轮预热＋3轮正式取均值；每批清空KV；同一个真实LoRA；greedy、固定输入/输出、ignore_eos=true。输入为重复filler且每请求前8个token不同。本表是整机输出token/s，包含排队、PD传输、Python客户端及原始证据写盘开销。三轮为独立突发批次，不代表长期稳态吞吐。TTFT/TPOT分位数合并正式轮次请求按nearest-rank计算；TPOT是SSE首尾之间按新增token数归一的均值，不是逐token网络事件间隔。','',
'|输入/输出|总并发|输出token/s均值|三轮范围|TTFT均值/P95秒|TPOT均值/P95毫秒|',
'|---|---:|---:|---:|---:|---:|']
for s in speed:
 L.append(f"|{s['input_tokens']}/{s['max_output_tokens']}|{s['concurrency']}|{s['throughput_mean']:.2f}|{min(s['throughput_rounds']):.2f}–{max(s['throughput_rounds']):.2f}|{s['ttft_s']['mean']:.3f}/{s['ttft_s']['p95']:.3f}|{s['tpot_s']['mean']*1000:.3f}/{s['tpot_s']['p95']*1000:.3f}|")
L+=['','逐轮数值见[throughput.csv](throughput.csv)。固定负载MTP接受率范围：'+f"{min(s['accept_rate'] for s in speed if s['accept_rate'] is not None):.2%}–{max(s['accept_rate'] for s in speed if s['accept_rate'] is not None):.2%}"+'。不能外推为真实业务容量，也未做八卡多副本合并部署的对照。','',
'自然文本单独使用原来的8个短中英文输入循环构造，单输入17–29 tokens，最大生成256，允许EOS；不是长文任务集。','',
'|总并发|输出token/s三轮均值|TTFT均值/P95秒|TPOT均值/P95毫秒|MTP接受率|','|---:|---:|---:|---:|---:|']
for s in natural:L.append(f"|{s['concurrency']}|{s['throughput_mean']:.2f}|{s['ttft_s']['mean']:.3f}/{s['ttft_s']['p95']:.3f}|{s['tpot_s']['mean']*1000:.3f}/{s['tpot_s']['p95']*1000:.3f}|{s['accept_rate']:.2%}|")
gpu={i:[] for i in range(8)};formal=[g for g in a['phases']['speed']['groups'] if g['label'].startswith('speed-')]
with tarfile.open(D/'speed-raw.tgz') as t:
 for line in t.extractfile('speed/gpu-metrics.jsonl'):
  x=json.loads(line)
  if not any(g['started_monotonic']<=x['monotonic']<=g['ended_monotonic'] for g in formal):continue
  for s in x['raw'].splitlines():
   v=[float(n.strip()) for n in s.split(',')];gpu[int(v[0])].append(v[1:])
L+=['','正式性能轮次内nvidia-smi采样汇总；显存为used MiB、功耗为采样均值/最大值，非设备功耗上限。短批次可能没有被指标采样捕捉，不能把观测到的0运行数理解为容量0。','',
'|GPU/角色|显存最大MiB|平均GPU-Util %|功耗均值/峰值W|温度峰值℃|','|---|---:|---:|---:|---:|']
for i,vs in gpu.items():
 L.append(f"|{i}/{'P' if i<4 else 'D'}|{max(v[0] for v in vs):.0f}|{statistics.mean(v[1] for v in vs):.2f}|{statistics.mean(v[3] for v in vs):.1f}/{max(v[3] for v in vs):.1f}|{max(v[4] for v in vs):.0f}|")
L+=['','## KV与长上下文','', '|实例|分配KV tokens|','|---|---:|']
for n,v in kv.items():L.append(f'|{n}|{v:,}|')
L+=['',f"四个D合计 **{sum(v for n,v in kv.items() if 'decode' in n):,} tokens**。每个D是独立池，不能给单请求合并成更长上下文；P池与D池也不相加。P转移后可回收，P池大小不是总decode驻留上限。",'',
'|输入/输出|总并发|整批完成秒|说明|','|---|---:|---:|---|']
for g in a['phases']['capacity']['groups']:L.append(f"|{g['input_tokens']}/{g['max_output_tokens']}|{g['concurrency']}|{g['elapsed_s']:.3f}|自然速度，无人工减速；容量组仅1轮|")
p=cap['peak_qualified_window']
L+=['',f"另外，128路×(131,072输入＋512输出)完成，4个D各32路。人为对D加入4秒/forward以观察驻留，合格窗口{len(cap['qualified_windows'])}次；峰值同一轮并发采样：运行请求{p['running_requests']:.0f}、活跃序列总长 **{p['active_seq_tokens']:,.0f}**、KV used **{p['kv_used_tokens']:,.0f}**，四个采样跨度{p['span_s']:.4f}秒。它是近同时的四实例采样，不是原子的全局快照；不累加不同时间的单实例峰值。这组包含人工减速，**绝不计入吞吐成绩**。减速已两次重置，结束后四对真实生成复测通过。",'',
'上限边界：262,012输入＋128输出在四对实例均完成；262,013＋128在全部8个实例均拒绝，校验总额为262,145（含4个MTP预留）。最初按262,016＋128发出的4个逻辑请求也被正确拒绝，原始失败、8条已完成容量请求和修正后的完整补测均保留。未测长文语义质量。','',
'## LoRA与数值一致性','',
'每实例16个GPU批身份槽位，base占一项；CPU缓存上限32。8个实例复制的是同一组32个身份，不能宣称256个独立训练LoRA。身份沿用同一真实rank16的A/B权重，alpha为1/32至32/32，是合成身份容量实验。每个D都实际完成32种身份请求；base＋16身份也都完成；靠分批和换入，不是同时放进16槽。', '',
'原槽位抽查12组独立/混流的32-token输出，11/12整段相同；slot-15、D1在第22个token分歧，原输入完全相同。针对该问题，另按相同128请求混流结构重放并收集logprobs：', '',
f"- 整段一致：{lp['whole_output_equal']}/{lp['total_comparisons']}。",
f"- Prefill同token：{lp['prefill']['count']}评分，mean/P99/max＝{lp['prefill']['mean_abs_nat']:.8f}/{lp['prefill']['p99_abs_nat']:.8f}/{lp['prefill']['max_abs_nat']:.8f} nat，超过原mean≤0.01/P99≤0.1阈值，**未通过**。",
f"- Decode相同前缀、相同评分token：{lp['decode_common_prefix']['count']}评分，mean/P99/max＝{lp['decode_common_prefix']['mean_abs_nat']:.8f}/{lp['decode_common_prefix']['p99_abs_nat']:.8f}/{lp['decode_common_prefix']['max_abs_nat']:.8f} nat；原mean≤0.01/P99≤0.1门槛结果：{lp['passes_old_common_prefix_gate']}。",'',
'上述decode评分遇首次生成分歧即停止，没有完成全部续写的teacher-forced验收。相同FP8配置的副本一致性检查：base/LoRA各覆盖16种P→D配对；另8个自然输入各跨4个D，64次生成、54组非自身对比的完整输出均相同，prefill/首token/同前缀decode差异均0。它不能替代独立merge对照；此前FP8 MTP开关及相对旧BF16的精度问题仍未闭环。','',
'## 基于本基线的优化顺序','',
'先解释混流的数值差异并补独立FP8 merge验收。性能优化保持4P4D/TP1作对照，每次只改一个变量，优先比较P端chunked-prefill=8192/16384/32768，观察32K负载的TTFT和吞吐，同时检查KV/SSM容量；再评估MTP参数及LoRA换入开销。需要用实际业务的输入/输出分布和延迟要求选择优化结果。','',
'32K输入、并发64时，三个正式轮次中每个D采样到的最高运行数为'+str(list(next(s for s in speed if s['input_tokens']=='32768' and s['concurrency']==64)['worker_max_running'].values())[4:])+'，低于每D分到的16路；首token等待显著、生成间隔约4.13ms，提示优先剖析prefill与传输。GPU利用率是采样指标，不等同于SM占用率，不能单独用它归因。','',
'## 复现与证据','',
'启动参数见8份launch-*.json，端口/物理GPU/全部P-D分发规则见topology.json。实验客户端直接调用两个角色的原生/generate；不是新增生产入口或故障切换方案。新实例持续运行，main适配器已恢复。旧6个实验容器已停止，switch-before.json保留恢复argv；恢复前需要先停止当前八个实例以释放显存。','',
'sources/保存启动、测量、调度、边界补测、驻留、精度重放和审计源码。复用原实验主目录中的模型、真实adapter、JIT模板、既有filler和自然输入contract；权重及编译缓存不入库，旧fp8-mtp/保留所依赖的原始模板及辅助函数。先完成smoke/speed/natural/slots；capacity初次上限拒绝后由capacity_pd4_r263.py完成正确边界补测；再运行residency_pd4_r262.py及precision_slots_pd4_r266.py；最后finalize/audit/export。所有阶段串行，不让测速与其它GPU测试重叠。','',
f"audit-r256.json从原始请求复核常规基线{a['verified_total_requests']}条生成；驻留审计另复核128条和原容量尝试8条；精度重放208条独立复核，结束后4条联合生成另存原始响应。累计5,980条完成的逻辑生成请求（含预热、重复与最终探针），不等同于独立真实模型质量样本。每个raw.tgz内含请求、P响应、D的SSE、时间戳和原始metrics；raw-files-sha256.json逐文件封存，manifest.json封存导出文件。精度重放数据见mixed-lora-logprob-comparison.json，208项引擎哈希见runtime-overlay-audit.json。"]
(D/'README.md').write_text('\n'.join(L)+'\n')
