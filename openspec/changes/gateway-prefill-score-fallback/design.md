## Context

当前 SGLang worker 已通过 `/v1/loads` 提供 core 指标和有限的 `prefill_queue` 聚合摘要；debug Gateway 已有后台 load poller、logical `prefill_members` 快照、Router reservation 和 predicted-TTFT 评分，但评分仍以 uncached token / fixed capacity 为主。Gateway 与 worker 之间的 load 请求可能因为认证、网络、worker 重启或 schema 差异失败，因此路由必须继续使用可解释的降级数据。

本变更只针对 debug Gateway 和其使用的 SGLang debug 分支。生产 `deploy-prod`、生产 Gateway 和 `macaron-llm-deploy` 不在迁移范围内。

## Goals / Non-Goals

**Goals:**

- 让 SGLang scheduler 暴露有界、带版本和 worker boot 标识的 waiting/running Prefill 快照。
- 让 Gateway 使用 `F_i(x)` 和 PD 的长度相关 `G_i(x) = F_i(x) / beta_i(x)` 计算 PrefillScore。
- 让路由热路径只读取后台缓存，不同步等待 `/v1/loads`。
- 在 enhanced snapshot、旧聚合字段、陈旧快照、Redis reservation、local reservation 和 fixed-capacity 之间逐级 fallback。
- 让 snapshot 与 reservation 能用同一个内部 request ID 去重，并暴露 load source、snapshot age 和 fallback reason。

**Non-Goals:**

- 不把 Decode load、KV transfer、bootstrap 或网络通信开销建模进 PrefillScore。
- 不修改生产 Gateway、`deploy-prod`、APIM/AFD、macaron-llm-deploy 或生产 worker membership。
- 不在本变更中实现完整的 worker stage event 总线；快照先采用有界轮询。
- 不把客户端 prompt 或其他敏感请求内容写入 load snapshot 或路由日志。

## Decisions

### 1. Extend `/v1/loads` rather than add a synchronous routing RPC

在现有 `/v1/loads` 增加可选 `prefill_work` section，保留旧 response 字段。快照包含 `schema_version`、`snapshot_id`、`generated_at_ms`、`worker_boot_id`、`detail_complete`、`truncated`，以及有界的 waiting/running 明细和 overflow summary。

Gateway 通过后台 poller 获取快照。选择请求时只读取内存中的 last-good snapshot；这样 load 接口网络抖动不会直接增加请求 TTFT。

备选方案是每次选路同步调用新的 estimate RPC，但它会把 worker 网络延迟直接放入路由关键路径，并在 worker 过载时形成反馈放大，因此不采用。

### 2. Use bounded request details with aggregate overflow

每个明细项只传内部 `request_id`、priority、uncached token 和运行中 Prefill 的 processed/current-chunk 位置。响应必须有固定项数或字节上限；超出部分按 priority + length bucket 聚合为 `overflow_summary`，并设置 `truncated=true`。

完整明细时 Gateway 计算：

```text
waiting_work = Σ F_i(waiting.uncached_tokens)
running_work = Σ [F_i(current_chunk_end) - F_i(processed)]
```

PD 使用 `G_i` 替换 `F_i`。不把聚合 token 当作一条长请求直接套非线性曲线。

### 3. Keep Prefill-only scoring and make PD preference explicit

Integrated worker 使用等待 work、运行 work 和新请求 `F_i(x)` 的和。PD worker 使用相同的 work 结构，但按长度桶应用 `beta_i(x)`。Decode 只通过现有 health/ready/routable 门禁，不进入分数。

所有分数保持毫秒量纲。曲线缺失时，profile 退化为 `x / fixed_capacity`；如果候选之间无法统一量纲，整池回到共同 fixed-capacity 模式。

### 4. Treat load transport failure separately from worker health failure

poller 为每个 worker 维护 `last_good_snapshot`、失败计数、schema/boot 校验状态和 snapshot age。load endpoint 超时但 health/probe 正常时，worker 不被自动摘除，只切换 fallback；health/probe 失败时沿用现有 fail-closed 路由门禁。

旧 boot 的快照在 `worker_boot_id` 改变后立即失效。陈旧快照只在可配置的 stale grace 内使用，并以 reservation 和年龄惩罚做保守修正；超过 grace 后不得伪装成实时负载。

### 5. Merge reservation conservatively and preserve provenance

同一 request ID 出现在 worker snapshot 与 Router reservation 中时只计一次。没有 request ID 的旧摘要无法逐请求去重，使用同量纲的 `max(snapshot_work, reservation_work)` 以覆盖轮询窗口而不整份双计。

每次决策记录 `load_source`、`snapshot_id`、`snapshot_age_ms`、`schema_version`、`detail_complete`、`truncated`、`fallback_reason` 和原始/有效 PrefillScore。

## Risks / Trade-offs

- **[Risk]** 逐请求快照增加 `/v1/loads` payload 和序列化成本。→ 使用固定项数/字节上限、overflow summary，并将明细作为可选 section。
- **[Risk]** 忽略通信和 bootstrap 会让 PrefillScore 系统性低估端到端 TTFT。→ 保留 actual TTFT 观测，按 profile/长度桶检查残差；不把残差伪装成精确通信模型。
- **[Risk]** stale snapshot 可能把已完成请求算作 work。→ stale grace、snapshot age 惩罚、boot ID 校验和 reservation max 合并，宁可短时保守少用该 worker。
- **[Risk]** beta 过大导致 PD 过载。→ beta 按长度桶配置并记录 effective score，通过 debug workload 校准上限。
- **[Risk]** enhanced schema 在旧 worker 上不可用。→ 保留 `/get_load`、旧 `/v1/loads` 聚合解析和 fixed-capacity fallback；schema 不兼容不能按零负载处理。

## Migration Plan

1. 在本分支增加 OpenSpec 对应测试和 SGLang enhanced snapshot 字段，先用本地 fake worker 验证 schema、截断和 fallback。
2. 构建 debug Gateway/worker artifact，在 `llm-gw-debug-0715` 的隔离 debug 服务上逐步启用 PrefillScore。
3. 先验证 fresh snapshot、旧聚合 fallback、load endpoint 超时、worker boot 变化和 Redis 不可用；再运行真实 workload/profile benchmark。
4. 回滚时切回 debug Gateway 之前的 binary/config，worker `/v1/loads` 新字段为可选，不需要修改旧 worker；不触碰生产服务。

## Open Questions

- 真实 debug worker 的请求 ID 是否能稳定贯穿 Gateway reservation、logical proxy 和 engine；若不能，需要使用内部 header/metadata 建立映射。
- 各 profile 的 `beta_i(x)` 初始值和上限需要真实 benchmark 后确定。
- overflow bucket 的代表 token 值应取中位数还是保守上界，需要用长短请求混合 workload 验证。
