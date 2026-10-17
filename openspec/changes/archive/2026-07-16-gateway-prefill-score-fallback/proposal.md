## Why

当前 Gateway 的 predicted-TTFT 只能使用聚合 uncached token 和固定 capacity，无法准确表达等待中的 Prefill work、正在执行的 chunk 剩余 work，以及不同机器在不同输入长度下的 Prefill 速度差异。与此同时，worker load 接口可能超时、返回旧 schema 或短暂不可达，不能让路由热路径同步等待，也不能把未知负载当成零。

本变更在 debug Gateway 上引入 PrefillScore 路由模型和 SGLang 有界 Prefill 快照，同时保证 load snapshot、Redis reservation 和 fixed-capacity 旧算法之间可以逐级 fallback。

## What Changes

- 扩展 SGLang `/v1/loads`，提供带 schema、snapshot、worker boot 标识的有界 waiting/running Prefill 明细和 overflow 摘要。
- 在 Gateway 后台轮询并缓存 worker load snapshot，路由热路径只使用 last-good 数据和 reservation，不同步等待 worker load 接口。
- 使用机器 profile 的 Prefill 曲线 `F_i(x)`，并支持 PD 按长度桶配置速度系数 `G_i(x) = F_i(x) / beta_i(x)`。
- 仅按 Prefill work 计算路由分数；不把 Decode load、KV 通信、bootstrap 或网络通信开销加入分数，Decode 只沿用健康/可路由性门禁。
- 以 request ID 合并 worker snapshot 与 Router reservation，避免重叠 work 双计。
- 增加 schema 不兼容、接口超时、陈旧 snapshot、Redis 不可用和 fixed-capacity 的多级 fallback，并为每次选路记录来源和降级原因。
- 保持 `/get_load` 和现有聚合 `/v1/loads` 字段兼容；enhanced 字段缺失时不影响旧 worker 参与 debug 路由。

## Capabilities

### New Capabilities
- `prefill-score-load-snapshot`: 提供 Prefill work 快照、PrefillScore 计算、reservation 合并和 load snapshot fallback。

### Modified Capabilities
- 无。

## Impact

- SGLang Python scheduler/load snapshot 与 `/v1/loads` 响应结构。
- `experimental/sgl-router` 的 load poller、worker snapshot 类型、predicted-TTFT/PrefillScore 策略和路由观测日志。
- Debug Gateway 编译产物和隔离 debug 服务；不修改 `deploy-prod`、生产 Gateway 或部署配置仓库。
- 新增单元、协议兼容、fallback 和路由选择测试；需要在真实 debug worker 上验证快照、失联 fallback 和请求路径。
