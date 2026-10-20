# Qwen3.8-27B-FP8、FP8 KV 与 MTP 实测

本轮按用户优先级暂停 DCP/PCP/DPA，先验证指定 FP8 配方。**启动、动态 LoRA、速度和容量测试完成；新配置未通过原精度阈值。** 原 BF16 1P1D 保持运行，最后联合生成复测通过。

## 配置与复现

官方模型 [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)，固定 revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。66 个 safetensors 分片共 30,866,866,928 bytes（30.87 GB）；1 个内置 MTP 层、22 个 MTP tensor 名称。官方原生上下文 262,144，扩展上限 1,000,000；本轮只配置原生长度。模型配置、文件列表与下载命令在 model-provenance/。

三组均为单实例 TP1；同一台 B300 主机的不同物理 GPU：

|组|物理 GPU|cache strategy|MTP|分配 KV tokens|
|---|---|---|---|---:|
|colocated，用户参数原样|2|extra_buffer_lazy|EAGLE 3/1/4|2,734,144|
|pd，兼容性调整|4=P、5=D|extra_buffer|EAGLE 3/1/4|P 1,169,728；D 4,758,336|
|off，对照|3|extra_buffer_lazy|关闭|1,207,296|

通用用户参数：

```bash
--model-path /work/models/Qwen3.8-27B-FP8 \
--kv-cache-dtype fp8_e4m3 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--mamba-full-memory-ratio 3.74 \
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--mamba-radix-cache-strategy extra_buffer_lazy \
--mamba-ssm-dtype bfloat16
```

当前 SGLang 在 PD 模式明确拒绝 `extra_buffer_lazy`，第一次原样 PD 启动失败已保留（model-provenance/startup-failed.json、final-state-and-logs.tgz）。PD 组只将该策略改为 `extra_buffer`。不绕过未实现的 lazy decode pool 校验。

其余设置：context=262144、mem-fraction=.80、max-running=64、chunked-prefill=8192、LoRA rank 上限32、GPU槽位16、CPU缓存上限32；实际 adapter 为真实 rank16。FP8 使用 Triton additive LoRA；旧 `merged_bf16` backend 明确拒绝量化和 speculative decoding。CUDA graphs、FP8 GEMM、attention 与 GDN 采用当前默认优化路径，不沿用旧 BF16 的 deterministic eager / torch GEMM 设置。移除了旧的 max-mamba-cache-size=256 显式上限，让 ratio 生效。

关闭 MTP 对照只去掉四个 MTP 参数，但自动 KV/SSM 分配会随之变化；不是相同 KV 配额的对照。速度负载的最长总 prompt 仅65,536，远低于各自池配额。与旧 BF16 的差异是整套配置的差异，不能归因于某一个优化。

复现顺序：先固定模型 revision、runtime image 和真实 adapter；按各 launch-*.json 的 Docker argv 在空闲 GPU 创建并启动容器；等 PD 两端日志出现 End of disaggregation warmup；运行 sources/measure_fp8_r239.py 的 smoke/speed/capacity/natural，slots 使用修正后的 measure_fp8_r244.py；只在速度测试结束后运行 residency_fp8_r241.py；最后运行 audit_fp8_r247.py。脚本保留原实验目录约定，所需 filler 可直接从原始请求取出；完整 token 输入均在各组 raw.tgz 内。模型和 adapter 权重不入库。runtime.json 和 runtime-overlay-audit-r248.json 固定镜像及26个fork覆盖文件的哈希；容器继承的 .git/OCI 标签仍指向上游，不能把它当作覆盖代码版本。

## 速度

输出 token/s；每项1次预热、3轮正式测试取算术均值。每批 flush KV，单 adapter，greedy、ignore_eos=true、固定输出长度，不强制生成内容。各组计时负载串行运行；包含请求排队、PD传输和原始SSE落盘成本。[逐轮CSV](throughput.csv) 保留波动。

|输入/输出|并发|单卡 MTP关闭|单卡 MTP开启|两卡 1P1D MTP开启|单卡 MTP开/关|
|---|---:|---:|---:|---:|---:|
|128/128|1|76.33|233.45|231.52|3.06x|
|128/128|32|1740.05|3586.85|3441.40|2.06x|
|128/128|64|2687.95|4459.27|4509.86|1.66x|
|2048/256|32|1173.93|1899.30|2088.22|1.62x|

该速度语料以重复 token filler 构造固定长度。正式测速 MTP 接受率约97%–100%，接受长度3.88–4.00；8个英文/中文短输入的自然生成抽查，接受率62.59%、接受长度2.84。因此这些吞吐是固定压力负载结果，不能直接当作线上任务吞吐。128输入下两卡PD没有稳定超过单卡；2048/256、并发32时，PD比单卡约快9.95%，但占两卡。未测两张卡各跑一个副本的总吞吐，不能宣布PD全局最优。

旧 BF16 1P1D 的128/128并发1/32/64为18.08/545.69/919.00 token/s，2048/256并发32为468.33；它使用另一套backend、graphs和缓存设置，仅作历史整套方案对照。

## KV 与上下文

单卡：2,734,144；PD：P 1,169,728、D 4,758,336，均为分配容量，不相加。**较小端的值也不是整个PD的总decode驻留上限**：P可在转移后回收，再为下一批请求prefill；P约束prefill阶段，D约束decode驻留，需分别实测。

两组都完成32768、131072、261632输入各加128输出的单请求；最长总长度261760，尚未测满262144。最长请求单卡16.04s、PD16.47s。两组32×(32768+1024)均完成；这是固定token压力，不证明长文任务质量。

PD补充驻留测试：32×(131072输入+512输出)全部完成。D端人为加2秒/forward以观察全批驻留，在两次合格快照后解除；独立原始指标复核发现8次整批驻留采样。同次峰值为32运行请求、活跃序列总长4,203,252、KV used 4,204,352 tokens，D池4,758,336。P池仍为1,169,728。该组256.10s含人工减速，**不计入性能成绩**；结束后两次null重置已核对。参见 audit-r247.json 的 residency-paced 与 pd-residency-paced-raw.tgz。

## LoRA 槽位

两组都成功缓存32个LoRA身份，并完成32身份各1请求（128/16）以及base+16 LoRA共17身份混流。抽查5个alpha版本在独立与32身份混流时的16-token输出完全相同。槽位测试只跑1轮，包含载入/换入，不能代替稳定混流吞吐基准。

`max-loras-per-batch=16` 是GPU批身份预算，base也占一个身份；`max-loaded-loras=32` 是CPU缓存驻留上限。32个身份通过排队、淘汰与换入完成，不是同时挤进16槽。第33次load成功并LRU淘汰一项，仍留32项；测试脚本最初错误预期拒绝，已修正，两个失败的客户端尝试保留。容量adapter共享同一真实A/B权重，alpha为1/32至32/32，是合成身份容量实验，不是32个独立训练模型验收。

## 精度

8个17–29-token输入、2轮×cold/hot，共32次自然生成，每次32输出token。prefill按实际输入token的原始logprob评分，另记录argmax；decode只比较相同已生成前缀且评分token相同的位置，遇到首次分歧即停止，未强制整个续写。

|对照|完整输出一致|prefill mean/P99|max-prefill|decode同前缀 mean/P99|
|---|---:|---:|---:|---:|
|FP8 MTP：PD vs单卡|32/32|0 / 0|0|0 / 0，992评分|
|同FP8：MTP开 vs关|12/32|0 / 0|0|0.01658 / 0.15384，716评分|
|FP8 MTP vs旧BF16独立merge|4/32|0.21033 / 2.49993|4.61670|未作全续写同前缀验收|

首token评分：FP8 PD/单卡/MTP关闭之间32个评分全0。原阈值mean≤0.01、P99≤0.1 nat；MTP开关的decode与相对旧BF16的prefill未通过。PD/单卡的700 prefill＋32首token＋992 decode共1724评分全0，只支持当前短样本中PD未引入可观察差异。相对旧merge包含权重/activation/KV/SSM/内核/LoRA backend变化；**不能解释为LoRA单项误差，也没有完成新FP8配置自身的独立merge精度验收**。下一步应先隔离这些数值变化并补任务质量评估；不能直接沿用旧BF16“精度已对齐”的结论。

Reasoning解析与工具调用均有真实OpenAI聊天响应：reasoning_content非空、最终答案42；tool_calls解析为get_weather(city=Beijing)。见parser-checks.json和final-state-and-logs.tgz。

## 证据

[audit-r247.json](audit-r247.json)独立复核单卡639、PD671、MTP关闭550，共1860个自然请求；最终另有旧BF16/新FP8的联合生成与parser冒烟。所有已完成phase均完整保留，请求输入、P响应、D/SSE事件、时间戳、原始metrics在对应raw.tgz；raw-files-sha256.json列出2000个原始文件哈希，manifest.json封存导出文件；未发布权重/JIT缓存。

原样PD的lazy策略拒绝，以及slots-attempt-r232（软链接路径不可见）/slots-attempt-r239（错误预期CPU缓存溢出应拒绝）保留。两次客户端问题修正后重跑；成功的速度/容量结果没有替换。四卡CP实验按用户要求暂停。
