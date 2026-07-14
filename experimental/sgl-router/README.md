# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker
pools come from either a static URL list or Kubernetes EndpointSlice
discovery.

## Building

```bash
cd experimental/sgl-router
cargo build --release
```

## Running

The router is configured entirely through CLI flags (run
`sgl-router --help` for the full list). It serves exactly one model, so
`--model-id` is required, along with exactly one discovery backend.
`--tokenizer-path` is optional: give it a local `tokenizer.json` path or a
HuggingFace repo id, and when omitted the router downloads the tokenizer
for `--model-id` from HuggingFace (honoring `HF_TOKEN` / `HF_HOME`).

Static worker list:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

ACA-style environment variables are also supported when the process starts
without CLI flags. `WORKER_URLS` has priority. If it is unset, the router can
load the static worker list from a macaron worker registry at startup:

```bash
MODEL_ID=zai-org/GLM-5.2-FP8 \
WORKER_REGISTRY_POOL=glm52-main \
WORKER_REGISTRY_APP_CONFIG_ENDPOINT=https://macaron-llm-deploy-prod.azconfig.io \
WORKER_REGISTRY_APP_CONFIG_KEY=macaron/prod/worker-registry/glm52/current \
WORKER_REGISTRY_APP_CONFIG_LABEL=prod \
sgl-router
```

Registry source options are mutually exclusive:

- `WORKER_REGISTRY_JSON`: inline registry JSON.
- `WORKER_REGISTRY_FILE`: local registry JSON file.
- `WORKER_REGISTRY_APP_CONFIG_ENDPOINT` + `WORKER_REGISTRY_APP_CONFIG_KEY`
  with optional `WORKER_REGISTRY_APP_CONFIG_LABEL`.

The App Configuration path authenticates with Azure managed identity. Set
`WORKER_REGISTRY_APP_CONFIG_MANAGED_IDENTITY_CLIENT_ID` when a user-assigned
identity should be used. `WORKER_REGISTRY_URL_SUFFIX` may append a common
static-discovery suffix such as `@tier=shared`; per-worker
`pool_url_suffixes` in the registry take precedence. This is startup discovery,
not hot reload: changing the registry requires a new revision/restart unless a
future runtime polling path is enabled.

### Dedicated Prefill/Decode Proxy

`pd_proxy` mode is for a stateless, independently replicated router in front of
one prefill/decode worker group. It requires static `WORKER_URLS`, authenticates
the upstream gateway with one `PD_PROXY_API_KEY`, and preserves the request
priority already assigned upstream:

```bash
ROUTER_MODE=pd_proxy \
MODEL_ID=zai-org/GLM-5.2-FP8 \
WORKER_URLS="http://prefill-a:30100 http://prefill-b:30100 http://decode-a:30200 http://decode-b:30200" \
PD_PROXY_API_KEY=... \
WORKER_INTROSPECT_KEY=... \
WORKER_BEARER_KEY=... \
LOAD_POLL_INTERVAL_SECS=1 \
WORKER_PROBE_TIMEOUT_SECS=3 \
sgl-router
```

`WORKER_PROBE_TIMEOUT_SECS` bounds each `/get_load` and `/health` request. It
defaults to 3 seconds; raise it only when a deployment's expected network or
worker response latency can legitimately exceed that budget.

The mode exposes Chat generation only; Completions, Messages, Responses, and
cache-flush routes are not registered. It also exposes a keyed, SGLang-compatible
`/get_load` endpoint so an outer gateway can register the deployment as an
`sglang_proxy`; the endpoint skips Decode workers whose latest load poll failed
and returns 503 when none have a usable snapshot. `/readyz` returns 200 only
when at least one healthy Prefill and one healthy Decode are available.
Stateful Responses must remain on the existing compatibility proxy until
cross-replica state has an external store.

Kubernetes EndpointSlice discovery:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --service-discovery \
  --service-discovery-namespace prod \
  --selector app=engines-qwen3
```

Omit `--service-discovery-namespace` to watch all namespaces (requires
cluster-wide RBAC). For prefill/decode disaggregation, replace `--selector`
with `--prefill-selector` and `--decode-selector`.

## External Queue Admission

External routers can opt in to fail-fast overload protection:

```bash
sgl-router \
  --external-queue-admission-enabled \
  --external-queue-admission-threshold 8 \
  ...
```

When enabled, generation requests are rejected with HTTP 429
`external_queue_overloaded` before policy selection only if every healthy,
priority-eligible worker has effective load greater than the threshold.
Equal-to-threshold is admitted. With load polling enabled, effective load uses
reported worker queue depth plus router-local pending reservations; otherwise
it uses router-local pending reservations only.

The feature is disabled by default. Enabling it on live ACA/APIM deployments
requires a separate rollout approval and configuration change.

## Cache-State Reconciliation

Worker-authoritative cache reconciliation is disabled by default. When enabled,
the Python ZMQ publisher maintains the routing-visible block union off the
scheduler thread, emits periodic XOR-SHA256 digests, and emits bounded snapshots
containing `(parent hash, block hash, storage media)` entries. Cache-state tracks
epoch and sequence continuity per worker DP rank; a gap or mismatch excludes only
that rank from cache-hit matches until a matching digest or verified snapshot
restores trust. Inference/load routing remains available.

Worker `--kv-events-config` JSON fields:

- `reconciliation_enabled` (default `false`)
- `reconciliation_digest_interval_s` (default `30`)
- `reconciliation_snapshot_interval_s` (default `600`; `0` disables snapshots)
- `reconciliation_snapshot_chunk_bytes` (default `262144`)
- `reconciliation_max_snapshot_entries` (default `2000000`)

Cache-event-agent controls:

- `CACHE_EVENT_AGENT_SINK_QUEUE_CAPACITY` (default `4096`)
- `CACHE_EVENT_AGENT_SINK_MAX_ATTEMPTS` (default `3`)
- `CACHE_EVENT_AGENT_SINK_RETRY_BACKOFF_MS` (default `100`)
- `CACHE_EVENT_AGENT_SINK_DELIVERY_TIMEOUT_MS` (default `10000`)
- `CACHE_EVENT_AGENT_MAX_SINK_PAYLOAD_BYTES` (default `1048576`)
- `CACHE_EVENT_AGENT_METRICS_BIND` (default `127.0.0.1:9898`)

Cache-state controls:

- `CACHE_STATE_RECONCILIATION_ENABLED` (default `false`)
- `CACHE_STATE_RECONCILIATION_MAX_WORKER_RANKS` (default `4096`)
- `CACHE_STATE_RECONCILIATION_MAX_SNAPSHOT_ENTRIES` (default `2000000`)
- `CACHE_STATE_RECONCILIATION_MAX_IN_PROGRESS_SNAPSHOT_ENTRIES` (default `4000000`)
- `CACHE_STATE_RECONCILIATION_MAX_SNAPSHOT_CHUNKS` (default `65536`)
- `CACHE_STATE_RECONCILIATION_DEDUPE_WINDOW` (default `8192`)
- `CACHE_STATE_KAFKA_APPLY_MAX_ATTEMPTS` (default `3`)
- `CACHE_STATE_KAFKA_APPLY_RETRY_BACKOFF_MS` (default `250`)

Roll out consumers and agents first while reconciliation remains disabled. Then
enable an isolated cache-state consumer group and one worker that has restarted
with reconciliation enabled. Enable authoritative production responses only
after every active worker rank emits epochs and a fault-injection test has shown
gap detection plus snapshot recovery. Roll back by disabling
`CACHE_STATE_RECONCILIATION_ENABLED` and worker emission; the legacy three-field
event batch remains unchanged.

## License

Apache-2.0.
