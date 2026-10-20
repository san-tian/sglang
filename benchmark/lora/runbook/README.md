# 新模型 MoL：跑通与精度对齐操作单

每行是实际执行顺序；同一格内从上到下执行。脚本来自飞书第6章，保留原始请求、日志和验收断言。模型开发适配完成后，按此流程验收。

- 三个终端使用同一 CUDA/Python 环境，共享 `/work/mol`；已有 CUDA 匹配的 SGLang 依赖。`01b` 的 `pip --no-deps` 只安装固定源码。
- 两张 GPU 各能放入一份 BF16/TP1 模型；准备 base、两份不同的非零 plain LoRA A/B，以及两份 merge 的磁盘空间。示例 rank≤32。
- 把本目录放到 `/work/mol/runbook`；命令从任意目录执行。服务前台运行并持续占用终端，检查客户端在终端3运行。
- 新架构先修改模型映射、前向与独立 merge；本文的目标模块/后端是当前 Qwen 示例。脚本使用固定服务提交 `19955ce786d7c573f0a48a77231d5f251866fb1c`，检查客户端来自 `63a980b42f89c327fcd76968af30cd3f809eecc9`。

| 步骤 | 在哪个终端执行什么 | 看到什么再继续 |
| --- | --- | --- |
| 1 准备 | 终端3：先执行下方权重链接命令；再依次 `bash /work/mol/runbook/01b-install.sh`、`01c-check-inputs.sh`、`01d-record-versions.sh`、`01e-hash-inputs.sh`（后3条同样带完整路径）。 | 输入检查通过；编辑 `/work/mol/evidence/versions.json`，填模型/tokenizer版本、镜像、官方原生及扩展总长度、RoPE和来源。保留输入 SHA256。已有正确链接时跳过01a；已有源码/证据时使用新的实验目录，不覆盖。 |
| 2 开发核对 | 终端3：`bash /work/mol/runbook/02a-export-mapping.sh`；填写 `evidence/target-map-A.csv` 与 `target-map-B.csv`；完成实际代码适配后执行 `bash /work/mol/runbook/02b-check-mapping.sh`。 | 两份CSV的 runtime_layer、packed_slice、tp_shard 都有准确含义；未融合/TP1填 none。未知键、坏shape必须拒绝；核对 FP32(B@A)×alpha/r 及融合切片。填表检查不代替代码正确性。 |
| 3 测试输入 | 终端3：`bash /work/mol/runbook/03a-make-prompts.sh`，然后 `bash /work/mol/runbook/03b-make-zero-and-corpus.sh`。有自备30行 prompts.txt 时跳过03a。 | `evidence/corpus.json` 恰好30条固定input_ids，每条2—3840 tokens；adapter-zero的B为零。 |
| 4 启动与冒烟 | 终端1：`bash /work/mol/runbook/04a-start-d.sh`，保持运行。终端3：`bash /work/mol/runbook/04b-smoke.sh`。 | D在GPU0/30000运行；base/A/B/zero各生成16 tokens，zero与base的ID和logprob一致，A/B/zero加载和卸载均成功。 |
| 5 生命周期 | 终端3：`bash /work/mol/runbook/05-lifecycle.sh`；D继续运行，期间独占adapter操作。 | lifecycle/summary.json 的 passed=true：100轮换权重、100次坏加载、108次LRU、256-token在途生成通过。 |
| 6 merge A | 终端2：`bash /work/mol/runbook/06a-merge-and-start-ma.sh`，保持运行。终端3：`bash /work/mol/runbook/06b-check-merge.sh`。 | M-A在GPU1/30002运行且不启用LoRA；merged-A/MERGE_COMPLETE存在，输入和输出分片哈希一致，有非零增量。D始终运行原始base。 |
| 7 A精度对拍 | 终端3依次：`bash /work/mol/runbook/07a-check-servers.sh`、`bash /work/mol/runbook/07b-compare-a.sh`、`bash /work/mol/runbook/07c-accept-a.sh`。 | D/M配置相同；8对短测后完成240对主测。prefill、首token、decode分开逐例检查：均差≤0.01 nat，P99≤0.1 nat；自身重复噪声也达标，保存最大差与argmax。 |
| 8 B精度对拍 | 终端2按Ctrl-C停止M-A并等退出，再执行 `bash /work/mol/runbook/08a-merge-and-start-mb.sh`。终端3：`bash /work/mol/runbook/08b-compare-and-accept-b.sh`。 | 服务确认为M-B，B用自己的独立merge参考完成240对，达到与A相同阈值。 |
| 9 混流 | 终端3：`bash /work/mol/runbook/09-mixed.sh`；D继续运行。 | base/A/B/zero共48混流请求；48个首token、720个decode评分逐例达标。 |
| 10 真正缓存命中 | 终端2按Ctrl-C停止M-B并等退出，再执行 `bash /work/mol/runbook/10a-restore-ma.sh`。终端3依次执行 `bash /work/mol/runbook/10b-check-restored-ma.sh`、`bash /work/mol/runbook/10c-cache-restore.sh`。 | 两端确认为D/base与M-A；cache-restore/summary.json的passed=true，冷cached_tokens=0、两端热cached_tokens>0，自然续写及logprob通过。 |
| 11 长上下文 | 首次终端3：`bash /work/mol/runbook/11-long-context.sh`，先验4096。下一档按下方重启示例执行。 | 每档10输入×3轮×冷热=60对，输入L−256、输出256；全prefill/首token/decode通过。4次越界400后，2次短请求成功。B、混流、真实缓存命中及长文任务质量在该档另验。 |
| 12 保存结论 | 终端3：`bash /work/mol/runbook/12a-acceptance-template.sh`；填写evidence/acceptance.csv。停止所有客户端，再分别在终端1/2按Ctrl-C停止D/M，确认日志停止，执行 `bash /work/mol/runbook/12b-freeze-evidence.sh`。 | 每个adapter/模块/rank/长度/TP-DP-DCP范围注明通过、失败或未测，并指向证据；生成SHA256SUMS.json。声明范围前两阶段通过后进入性能优化。 |

首次链接输入，仅替换三个绝对路径；已有正确的 `/work/mol/base`、`adapter-A`、`adapter-B` 时跳过：

```bash
BASE_SRC='/绝对路径/基座目录' \
ADAPTER_A_SRC='/绝对路径/adapter-A目录' \
ADAPTER_B_SRC='/绝对路径/adapter-B目录' \
bash /work/mol/runbook/01a-link-inputs.sh
```

第1步安装后，另外三个命令完整写法：

```bash
bash /work/mol/runbook/01c-check-inputs.sh
bash /work/mol/runbook/01d-record-versions.sh
# 用编辑器填写 /work/mol/evidence/versions.json
bash /work/mol/runbook/01e-hash-inputs.sh
```

第11步扩大长度示例（8192总长）：先在终端1/2分别Ctrl-C停止D/M并等退出，再各运行对应命令，服务就绪后终端3运行长测。仅变更环境变量不会修改已运行服务。

```bash
# 终端1，保持运行
MOL_CONTEXT_LENGTH=8192 bash /work/mol/runbook/04a-start-d.sh
# 终端2，保持运行
MOL_CONTEXT_LENGTH=8192 bash /work/mol/runbook/10a-restore-ma.sh
# 终端3，确认两端启动成功后
L=8192 bash /work/mol/runbook/11-long-context.sh
```

扩展长度需要RoPE时，先按官方配置同步修改D/M启动脚本，再重启；档位以上述versions.json记录的官方上限为准。当前Qwen用4096/8192/32768/65536/131072/262144/524288/1000000，最后两档需要第1章对应YaRN配置。只增加context-length不能代替该配置。长测默认保留全prefill；OOM保留失败，跳过prefill仅作诊断。

命令失败就停在当前步，查看同名证据目录和D/M日志；修复后用新输出目录重跑。第2步需要开发适配，脚本导出表格不会自动实现新模型。完整逐步说明和失败定位仍见[飞书第6章](https://neocloud.feishu.cn/docx/W4NLdR7bhoHXL2xBcQDc8NeKnqd#doxcnI1FNJORaXdpbz4mS5Xc3Qe)。

`manifest.json`记录26份脚本的SHA256、来源文档版本与block。`syntax-check.json`仅记录shell语法和内嵌Python编译检查；本操作单整理没有新增GPU实验结果。
