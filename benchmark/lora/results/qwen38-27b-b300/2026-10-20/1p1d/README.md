# 两卡 1P1D 实测

主方案：P GPU0/TP1＋D GPU1/TP1；原始 base 动态挂载真实 rank16 LoRA。实验机独占两张 B300，其余六卡空闲。当前保持16 GPU槽位／最多登记32 adapters，两端各加载main；32 GPU槽位／64登记是另外完成的工作点。两端槽位与KV容量均不能相加。

官方Qwen3.8-27B模型卡：原生上下文262144，扩展上限1000000（https://huggingface.co/Qwen/Qwen3.8-27B）。本地config与官方逐字节一致；沿用qwen3_5架构字段。官方标称27B，本地完整checkpoint为27.781B个BF16元素（含视觉/MTP），18个权重分片共55.56GB/51.75GiB；KV/SSM/LoRA缓存另计。见model-basics-r187.json与official-model-reference-r187.json。

模型目录名 Qwen3.8-27B，实际config为Qwen3_5ForConditionalGeneration/qwen3_5_text，dense，非MoE。本轮为BF16 deterministic eager、merged_bf16、triton attention、torch GEMM、36GiB/role active投影缓存、关闭CUDA graph、NIXL/UCX、mem-fraction .80。两容器均可见GPU0/1，角色分别用base-gpu-id 0/1，避免CUDA枚举不一致。

## 速度

输出token/s，投影缓存预热、每批flush KV，1次预热＋3轮正式测试。单adapter greedy生成，ignore_eos=true固定长度，不强制token内容；D/M分别运行、同两卡与相同KV token配额。包含PD传输、排队及原始响应采集开销。

| 输入/输出 | 并发 | D tok/s | M tok/s | D/M | D TTFT P95(s) | D TPOT P95(ms) |
|---|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 18.08 | 20.17 | 89.63% | 0.110 | 55.80 |
| 128/128 | 32 | 545.69 | 593.62 | 91.92% | 0.365 | 56.10 |
| 128/128 | 64 | 919.00 | 1128.72 | 81.42% | 4.810 | 56.35 |
| 2048/256 | 1 | 18.29 | 20.06 | 91.21% | 0.115 | 54.52 |
| 2048/256 | 32 | 468.33 | 507.37 | 92.31% | 3.231 | 55.81 |
| 2048/256 | 64 | 771.40 | 845.03 | 91.29% | 6.396 | 58.63 |


短输入c64动态正式三轮为687.67/1034.95/1034.38，均值919.00；存在轮次波动，全部保留。历史TP1的KV配额更小，不能将跨组差值归因于PD。没有完成相同两卡双TP1副本对照，不能宣称1P1D为全局最优。详见[吞吐与P95 CSV](throughput-1p1d.csv)。

## KV与LoRA槽位

| GPU槽位/role | 登记上限/role | P分配KV tokens | D分配KV tokens |
|---:|---:|---:|---:|
|16|32|1131516|1131612|
|32|64|797660|797660|


恢复后的当前实际分配：P 1,131,452 / D 1,131,484 tokens；重启时自动分配有小幅浮动。主测速D/M使用的是表中初始配额。


接近池容量的补测：32×(32768输入＋2048输出)全部完成，总prompt 1048576 tokens。最多观察到30路同时运行；同次KV峰值1080039 tokens、运行序列总长1014473 tokens。32路整批同时驻留采样数为0，原32路检测项未通过，失败记录保留。批次耗时237.75秒。[原始响应及观察值复核](1p1d-KV-observed-audit-r185.json)通过不代表原检测项通过。KV used可能含队列/缓存分配，运行序列总长单独列出；这是多请求合计，非单请求百万上下文。


配对池受较小端限制，分配值不是满池压力验收。主方案32×(8192输入＋512输出)全部完成；同次decode采样峰值272903 KV tokens、32运行请求，22个采样观察到整批prompt驻留。32K输入＋512输出另已完成1路。本轮context配置262144（输入＋输出），尚未在此拓扑验收满上下文质量。Mamba/SSM另占内存：conv约0.71GiB、SSM约36.14GiB/role；不计入KV tokens。

槽位池按max rank32预留，实测adapter为rank16。32/64个容量adapter是同一真实A/B权重的不同alpha版本，支持性验证不等于32/64个独立训练模型的质量验收。base混流占一个批身份；16槽位一批支持15 LoRA＋base，32槽位支持31 LoRA＋base。更多身份请求可排队/换入。两端分别成功登记32/64，且混流超过池容量及base混流均实际完成。

| GPU槽位/登记 | 混流LoRA身份 | 输入/输出 | 输出tok/s，两轮均值 |
|---|---:|---|---:|
|16/32|1|128/32|16.47|
|16/32|4|128/32|19.72|
|16/32|8|128/32|18.48|
|16/32|16|128/32|17.60|
|16/32|32|128/32|16.94|
|32/64|1|128/32|16.75|
|32/64|16|128/32|17.02|
|32/64|32|128/32|16.99|
|32/64|64|128/32|16.15|


多LoRA吞吐明显下降；该组输出长度仅32，与主测速矩阵不同，包含可能的缓存填充/换入成本。源码run_merged_linear按adapter逐组调用F.linear，缓存miss时重新merge；计算路径已确认，但逐组GEMM、缓存换入和merge各自的耗时占比尚未profile。增加槽位不等于吞吐增加。详见[逐轮槽位CSV](lora-slots-1p1d.csv)。

## 精度与证据

动态D vs独立离线merge M：582/582个正式自然输出序列一致。短样本teacher forcing：8例×2轮×cold/hot＝32对，32输出token；原始强制之前的prefill/首token/decode评分分别如下。

- prefill: 700评分，mean 0 /P99 0 /max 0 nat。

- prefill_last: 32评分，mean 0 /P99 0 /max 0 nat。

- decode: 992评分，mean 0 /P99 0 /max 0 nat。


输入长度17–29，仅为本拓扑回归验证，不能替代完整长文数值或任务质量验收。阈值沿用mean≤0.01/P99≤0.1 nat。动态与原始base权重不同；本轮未给base任务准确率增益。

[主原始审计](1p1d-main-audit-r171.json)、[32槽位审计](1p1d-slot-sweep-audit-r172.json)、[最终健康及配置](final-live-state-r175.json)、[全部原始文件SHA256](raw-files-sha256.json)与各组tgz保留每个自然请求P响应/D SSE、时间戳、分配日志、驻留指标及数值capture。R161测试脚本复制临时socket失败，未启动merge模型；失败证据保留，R168修复缓存复制后重新完整执行，无删改原始测速。

全局多卡优化目标未完成；旧4P4D/DP8工作按用户优先级暂缓。当前结果支持暂用1P1D，后续重点为多LoRA混流性能与正式负载/SLO验证。
