use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use axum::extract::State;
use axum::routing::get;
use axum::Router;
use clap::Parser;
use reqwest::Client;
use sgl_router::cache_event_stream::{
    KafkaKvEventProducer, KafkaKvEventStreamConfig, KvEventStreamRecord, LocalKvEventStream,
    LocalKvEventStreamConfig,
};
use sgl_router::cache_state::CacheStateKvEventsRequest;
use sgl_router::policies::kv_events::tree::KvWorkerId;
use sgl_router::policies::kv_events::wire::decode_event_batch;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use zeromq::{Socket, SocketRecv, SubSocket};

const END_SEQ_SENTINEL: i64 = -1;

#[derive(Debug, Parser)]
#[command(about = "Forward local SGLang KV events to a remote cache-state service")]
struct Args {
    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_CACHE_STATE_URL",
        required_unless_present_any = ["stream_path", "kafka_bootstrap_servers"]
    )]
    cache_state_url: Option<String>,

    #[arg(long, env = "CACHE_EVENT_AGENT_STREAM_PATH")]
    stream_path: Option<PathBuf>,

    #[arg(long, env = "CACHE_EVENT_AGENT_STREAM_RETENTION_SECS")]
    stream_retention_secs: Option<u64>,

    #[arg(long, env = "CACHE_EVENT_AGENT_STREAM_MAX_BYTES")]
    stream_max_bytes: Option<u64>,

    #[arg(long, env = "CACHE_EVENT_AGENT_KAFKA_BOOTSTRAP_SERVERS")]
    kafka_bootstrap_servers: Option<String>,

    #[arg(long, env = "CACHE_EVENT_AGENT_KAFKA_TOPIC")]
    kafka_topic: Option<String>,

    #[arg(long, env = "CACHE_EVENT_AGENT_KAFKA_USERNAME")]
    kafka_username: Option<String>,

    #[arg(long, env = "CACHE_EVENT_AGENT_KAFKA_PASSWORD")]
    kafka_password: Option<String>,

    #[arg(long, env = "CACHE_EVENT_AGENT_KAFKA_CLIENT_ID")]
    kafka_client_id: Option<String>,

    #[arg(long, env = "CACHE_EVENT_AGENT_WORKER_URL")]
    worker_url: String,

    #[arg(long, env = "CACHE_EVENT_AGENT_MODEL_ID")]
    model_id: String,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_ENDPOINT_HOST",
        default_value = "127.0.0.1"
    )]
    endpoint_host: String,

    #[arg(long, env = "CACHE_EVENT_AGENT_PORT_BASE", default_value_t = 5557)]
    port_base: u16,

    #[arg(long, env = "CACHE_EVENT_AGENT_DP_SIZE", default_value_t = 1)]
    dp_size: u32,

    #[arg(long, env = "CACHE_EVENT_AGENT_TOPIC", default_value = "")]
    topic: String,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_POST_TIMEOUT_MS",
        default_value_t = 2000
    )]
    post_timeout_ms: u64,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_SINK_QUEUE_CAPACITY",
        default_value_t = 4096
    )]
    sink_queue_capacity: usize,

    #[arg(long, env = "CACHE_EVENT_AGENT_SINK_MAX_ATTEMPTS", default_value_t = 3)]
    sink_max_attempts: u32,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_SINK_RETRY_BACKOFF_MS",
        default_value_t = 100
    )]
    sink_retry_backoff_ms: u64,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_SINK_DELIVERY_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    sink_delivery_timeout_ms: u64,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_MAX_SINK_PAYLOAD_BYTES",
        default_value_t = 1024 * 1024
    )]
    max_sink_payload_bytes: usize,

    #[arg(
        long,
        env = "CACHE_EVENT_AGENT_METRICS_BIND",
        default_value = "127.0.0.1:9898"
    )]
    metrics_bind: SocketAddr,

    #[arg(long, env = "CACHE_EVENT_AGENT_LOG_LEVEL", default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&args.log_level)),
        )
        .with_target(true)
        .json()
        .init();

    if args.dp_size == 0 {
        return Err(anyhow!("--dp-size must be greater than 0"));
    }
    if args.sink_queue_capacity == 0
        || args.sink_max_attempts == 0
        || args.sink_delivery_timeout_ms == 0
        || args.max_sink_payload_bytes == 0
    {
        return Err(anyhow!(
            "sink queue, attempts, delivery timeout, and payload limit must be greater than zero"
        ));
    }
    let sinks = build_event_sinks(&args)?;
    let client = Client::builder()
        .timeout(Duration::from_millis(args.post_timeout_ms))
        .build()
        .context("build HTTP client")?;
    let cache_state_api_token = std::env::var("CACHE_STATE_API_TOKEN")
        .ok()
        .filter(|s| !s.is_empty());
    let cancel = CancellationToken::new();
    install_signal_handlers(cancel.clone())?;
    let metrics = Arc::new(AgentMetrics::default());
    let metrics_handle =
        spawn_metrics_server(args.metrics_bind, Arc::clone(&metrics), cancel.clone()).await?;
    let delivery_config = DeliveryConfig {
        queue_capacity: args.sink_queue_capacity,
        max_attempts: args.sink_max_attempts,
        retry_backoff: Duration::from_millis(args.sink_retry_backoff_ms),
        delivery_timeout: Duration::from_millis(args.sink_delivery_timeout_ms),
    };

    info!(
        worker_url = %args.worker_url,
        model_id = %args.model_id,
        endpoint_host = %args.endpoint_host,
        port_base = args.port_base,
        dp_size = args.dp_size,
        sinks = %sinks.iter().map(EventSink::name).collect::<Vec<_>>().join(","),
        metrics_bind = %args.metrics_bind,
        sink_queue_capacity = args.sink_queue_capacity,
        sink_max_attempts = args.sink_max_attempts,
        max_sink_payload_bytes = args.max_sink_payload_bytes,
        "cache-event-agent starting",
    );

    let mut handles = Vec::new();
    for dp_rank in 0..args.dp_size {
        let Some(port) = (u32::from(args.port_base) + dp_rank)
            .try_into()
            .ok()
            .map(|p: u16| p)
        else {
            warn!(
                dp_rank,
                port_base = args.port_base,
                "skipping dp rank whose ZMQ port overflows u16"
            );
            continue;
        };
        let task = AgentTask {
            client: client.clone(),
            sinks: sinks.clone(),
            worker_url: args.worker_url.clone(),
            model_id: args.model_id.clone(),
            endpoint: format!("tcp://{}:{}", args.endpoint_host, port),
            topic: args.topic.clone(),
            dp_rank,
            cache_state_api_token: cache_state_api_token.clone(),
            delivery_config,
            max_sink_payload_bytes: args.max_sink_payload_bytes,
            metrics: Arc::clone(&metrics),
            cancel: cancel.clone(),
        };
        handles.push(tokio::spawn(task.run()));
    }

    if handles.is_empty() {
        cancel.cancel();
        let _ = metrics_handle.await;
        return Err(anyhow!("no subscriber tasks were started"));
    }
    cancel.cancel();
    let _ = metrics_handle.await;
    for handle in handles {
        if let Err(err) = handle.await {
            error!(error = %err, "subscriber task panicked");
        }
    }
    info!("cache-event-agent stopped");
    Ok(())
}

#[derive(Debug, Default)]
struct AgentMetrics {
    publisher_sequence_gaps: AtomicU64,
    oversized_payloads: AtomicU64,
    queue_drops_http: AtomicU64,
    queue_drops_stream: AtomicU64,
    queue_drops_kafka: AtomicU64,
    delivery_failures_http: AtomicU64,
    delivery_failures_stream: AtomicU64,
    delivery_failures_kafka: AtomicU64,
    deliveries_http: AtomicU64,
    deliveries_stream: AtomicU64,
    deliveries_kafka: AtomicU64,
}

impl AgentMetrics {
    fn queue_drop(&self, kind: SinkKind) {
        self.counter(kind, MetricKind::QueueDrop)
            .fetch_add(1, Ordering::Relaxed);
    }

    fn delivery_failure(&self, kind: SinkKind) {
        self.counter(kind, MetricKind::DeliveryFailure)
            .fetch_add(1, Ordering::Relaxed);
    }

    fn delivery_success(&self, kind: SinkKind) {
        self.counter(kind, MetricKind::DeliverySuccess)
            .fetch_add(1, Ordering::Relaxed);
    }

    fn counter(&self, kind: SinkKind, metric: MetricKind) -> &AtomicU64 {
        match (kind, metric) {
            (SinkKind::Http, MetricKind::QueueDrop) => &self.queue_drops_http,
            (SinkKind::Stream, MetricKind::QueueDrop) => &self.queue_drops_stream,
            (SinkKind::Kafka, MetricKind::QueueDrop) => &self.queue_drops_kafka,
            (SinkKind::Http, MetricKind::DeliveryFailure) => &self.delivery_failures_http,
            (SinkKind::Stream, MetricKind::DeliveryFailure) => &self.delivery_failures_stream,
            (SinkKind::Kafka, MetricKind::DeliveryFailure) => &self.delivery_failures_kafka,
            (SinkKind::Http, MetricKind::DeliverySuccess) => &self.deliveries_http,
            (SinkKind::Stream, MetricKind::DeliverySuccess) => &self.deliveries_stream,
            (SinkKind::Kafka, MetricKind::DeliverySuccess) => &self.deliveries_kafka,
        }
    }

    fn render(&self) -> String {
        format!(
            concat!(
                "# TYPE sgl_router_cache_event_agent_publisher_sequence_gaps_total counter\n",
                "sgl_router_cache_event_agent_publisher_sequence_gaps_total {}\n",
                "# TYPE sgl_router_cache_event_agent_oversized_payloads_total counter\n",
                "sgl_router_cache_event_agent_oversized_payloads_total {}\n",
                "# TYPE sgl_router_cache_event_agent_sink_queue_drops_total counter\n",
                "sgl_router_cache_event_agent_sink_queue_drops_total{{sink=\"http\"}} {}\n",
                "sgl_router_cache_event_agent_sink_queue_drops_total{{sink=\"stream\"}} {}\n",
                "sgl_router_cache_event_agent_sink_queue_drops_total{{sink=\"kafka\"}} {}\n",
                "# TYPE sgl_router_cache_event_agent_sink_delivery_failures_total counter\n",
                "sgl_router_cache_event_agent_sink_delivery_failures_total{{sink=\"http\"}} {}\n",
                "sgl_router_cache_event_agent_sink_delivery_failures_total{{sink=\"stream\"}} {}\n",
                "sgl_router_cache_event_agent_sink_delivery_failures_total{{sink=\"kafka\"}} {}\n",
                "# TYPE sgl_router_cache_event_agent_sink_deliveries_total counter\n",
                "sgl_router_cache_event_agent_sink_deliveries_total{{sink=\"http\"}} {}\n",
                "sgl_router_cache_event_agent_sink_deliveries_total{{sink=\"stream\"}} {}\n",
                "sgl_router_cache_event_agent_sink_deliveries_total{{sink=\"kafka\"}} {}\n",
            ),
            self.publisher_sequence_gaps.load(Ordering::Relaxed),
            self.oversized_payloads.load(Ordering::Relaxed),
            self.queue_drops_http.load(Ordering::Relaxed),
            self.queue_drops_stream.load(Ordering::Relaxed),
            self.queue_drops_kafka.load(Ordering::Relaxed),
            self.delivery_failures_http.load(Ordering::Relaxed),
            self.delivery_failures_stream.load(Ordering::Relaxed),
            self.delivery_failures_kafka.load(Ordering::Relaxed),
            self.deliveries_http.load(Ordering::Relaxed),
            self.deliveries_stream.load(Ordering::Relaxed),
            self.deliveries_kafka.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum MetricKind {
    QueueDrop,
    DeliveryFailure,
    DeliverySuccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SinkKind {
    Http,
    Stream,
    Kafka,
}

impl SinkKind {
    fn label(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Stream => "stream",
            Self::Kafka => "kafka",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DeliveryConfig {
    queue_capacity: usize,
    max_attempts: u32,
    retry_backoff: Duration,
    delivery_timeout: Duration,
}

async fn spawn_metrics_server(
    bind: SocketAddr,
    metrics: Arc<AgentMetrics>,
    cancel: CancellationToken,
) -> Result<JoinHandle<()>> {
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind cache-event-agent metrics at {bind}"))?;
    let app = Router::new()
        .route("/metrics", get(agent_metrics))
        .with_state(metrics);
    Ok(tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app)
            .with_graceful_shutdown(cancel.cancelled_owned())
            .await
        {
            error!(error = %err, "cache-event-agent metrics server stopped with error");
        }
    }))
}

async fn agent_metrics(State(metrics): State<Arc<AgentMetrics>>) -> String {
    metrics.render()
}

#[derive(Clone)]
enum EventSink {
    Http {
        ingest_url: String,
        display_url: String,
    },
    Stream(LocalKvEventStream),
    Kafka(KafkaKvEventProducer),
    #[cfg(test)]
    Test {
        name: &'static str,
        kind: SinkKind,
        delay: Duration,
        delivered: Arc<AtomicU64>,
    },
}

impl EventSink {
    fn name(&self) -> &str {
        match self {
            Self::Http { display_url, .. } => display_url,
            Self::Stream(_) => "local-event-stream",
            Self::Kafka(producer) => producer.topic(),
            #[cfg(test)]
            Self::Test { name, .. } => name,
        }
    }

    fn kind(&self) -> SinkKind {
        match self {
            Self::Http { .. } => SinkKind::Http,
            Self::Stream(_) => SinkKind::Stream,
            Self::Kafka(_) => SinkKind::Kafka,
            #[cfg(test)]
            Self::Test { kind, .. } => *kind,
        }
    }
}

fn build_event_sinks(args: &Args) -> Result<Vec<EventSink>> {
    let mut sinks = Vec::new();
    if let Some(base_url) = args
        .cache_state_url
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let base_url = base_url.trim_end_matches('/').to_string();
        sinks.push(EventSink::Http {
            ingest_url: format!("{base_url}/v1/cache_state/kv_events"),
            display_url: base_url,
        });
    }
    if let Some(stream_path) = args.stream_path.clone() {
        sinks.push(EventSink::Stream(LocalKvEventStream::new(
            LocalKvEventStreamConfig {
                path: stream_path,
                retention_secs: args.stream_retention_secs,
                max_bytes: args.stream_max_bytes,
            },
        )));
    }
    if let Some(bootstrap_servers) = args
        .kafka_bootstrap_servers
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let topic = args
            .kafka_topic
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("--kafka-topic is required when Kafka is enabled"))?
            .to_string();
        let producer = KafkaKvEventProducer::new(KafkaKvEventStreamConfig {
            bootstrap_servers: bootstrap_servers.to_string(),
            topic,
            username: args.kafka_username.clone(),
            password: args.kafka_password.clone(),
            client_id: args.kafka_client_id.clone(),
            consumer_group: None,
            auto_offset_reset: "latest".to_string(),
        })?;
        sinks.push(EventSink::Kafka(producer));
    }
    if sinks.is_empty() {
        return Err(anyhow!(
            "configure at least one sink: cache-state URL, stream path, or Kafka"
        ));
    }
    Ok(sinks)
}

#[derive(Clone)]
struct SinkDelivery {
    record: Arc<KvEventStreamRecord>,
    n_events: usize,
}

#[derive(Clone)]
struct SinkDispatcher {
    kind: SinkKind,
    name: String,
    tx: mpsc::Sender<SinkDelivery>,
    metrics: Arc<AgentMetrics>,
}

impl SinkDispatcher {
    fn try_send(&self, delivery: SinkDelivery) {
        match self.tx.try_send(delivery) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(delivery)) => {
                self.metrics.queue_drop(self.kind);
                warn!(
                    sink = %self.name,
                    sink_kind = self.kind.label(),
                    dp_rank = delivery.record.dp_rank,
                    seq = delivery.record.seq,
                    "sink queue full; dropping KV event record"
                );
            }
            Err(mpsc::error::TrySendError::Closed(delivery)) => {
                self.metrics.delivery_failure(self.kind);
                warn!(
                    sink = %self.name,
                    sink_kind = self.kind.label(),
                    dp_rank = delivery.record.dp_rank,
                    seq = delivery.record.seq,
                    "sink worker closed; dropping KV event record"
                );
            }
        }
    }
}

fn spawn_sink_worker(
    sink: EventSink,
    client: Client,
    cache_state_api_token: Option<String>,
    config: DeliveryConfig,
    metrics: Arc<AgentMetrics>,
    cancel: CancellationToken,
) -> (SinkDispatcher, JoinHandle<()>) {
    let kind = sink.kind();
    let name = sink.name().to_string();
    let (tx, rx) = mpsc::channel(config.queue_capacity);
    let dispatcher = SinkDispatcher {
        kind,
        name: name.clone(),
        tx,
        metrics: Arc::clone(&metrics),
    };
    let handle = tokio::spawn(run_sink_worker(
        sink,
        client,
        cache_state_api_token,
        config,
        metrics,
        cancel,
        rx,
    ));
    (dispatcher, handle)
}

async fn run_sink_worker(
    sink: EventSink,
    client: Client,
    cache_state_api_token: Option<String>,
    config: DeliveryConfig,
    metrics: Arc<AgentMetrics>,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<SinkDelivery>,
) {
    let kind = sink.kind();
    let name = sink.name().to_string();
    loop {
        let delivery = tokio::select! {
            _ = cancel.cancelled() => return,
            delivery = rx.recv() => match delivery {
                Some(delivery) => delivery,
                None => return,
            },
        };
        let mut delivered = false;
        for attempt in 1..=config.max_attempts {
            let result = tokio::time::timeout(
                config.delivery_timeout,
                deliver_once(&sink, &client, cache_state_api_token.as_deref(), &delivery),
            )
            .await;
            match result {
                Ok(Ok(())) => {
                    metrics.delivery_success(kind);
                    debug!(
                        sink = %name,
                        sink_kind = kind.label(),
                        dp_rank = delivery.record.dp_rank,
                        seq = delivery.record.seq,
                        n_events = delivery.n_events,
                        attempt,
                        "delivered KV event record"
                    );
                    delivered = true;
                    break;
                }
                Ok(Err(err)) => {
                    warn!(
                        sink = %name,
                        sink_kind = kind.label(),
                        dp_rank = delivery.record.dp_rank,
                        seq = delivery.record.seq,
                        attempt,
                        max_attempts = config.max_attempts,
                        error = %err,
                        "sink delivery attempt failed"
                    );
                }
                Err(_) => {
                    warn!(
                        sink = %name,
                        sink_kind = kind.label(),
                        dp_rank = delivery.record.dp_rank,
                        seq = delivery.record.seq,
                        attempt,
                        max_attempts = config.max_attempts,
                        timeout_ms = config.delivery_timeout.as_millis(),
                        "sink delivery attempt timed out"
                    );
                }
            }
            if attempt < config.max_attempts {
                let multiplier = 1u32 << attempt.saturating_sub(1).min(10);
                let delay = config.retry_backoff.saturating_mul(multiplier);
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
        if !delivered {
            metrics.delivery_failure(kind);
            error!(
                sink = %name,
                sink_kind = kind.label(),
                dp_rank = delivery.record.dp_rank,
                seq = delivery.record.seq,
                "dropping KV event record after bounded sink retries"
            );
        }
    }
}

async fn deliver_once(
    sink: &EventSink,
    client: &Client,
    cache_state_api_token: Option<&str>,
    delivery: &SinkDelivery,
) -> Result<()> {
    match sink {
        EventSink::Http { ingest_url, .. } => {
            let record = &delivery.record;
            let req = CacheStateKvEventsRequest {
                model_id: record.model_id.clone(),
                worker_url: record.worker_url.clone(),
                dp_rank: record.dp_rank,
                seq: record.seq,
                payload_b64: record.payload_b64.clone(),
            };
            let mut post = client.post(ingest_url);
            if let Some(token) = cache_state_api_token {
                post = post.bearer_auth(token);
            }
            let response = post.json(&req).send().await?;
            if !response.status().is_success() {
                return Err(anyhow!(
                    "cache-state returned HTTP status {}",
                    response.status()
                ));
            }
            Ok(())
        }
        EventSink::Stream(stream) => {
            let stats = stream.append(&delivery.record)?;
            debug!(
                stream_records = stats.records_after_compaction,
                stream_bytes = stats.bytes_after_compaction,
                "appended KV event record to local stream"
            );
            Ok(())
        }
        EventSink::Kafka(producer) => producer.send(&delivery.record).await,
        #[cfg(test)]
        EventSink::Test {
            delay, delivered, ..
        } => {
            tokio::time::sleep(*delay).await;
            delivered.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
}

struct AgentTask {
    client: Client,
    sinks: Vec<EventSink>,
    worker_url: String,
    model_id: String,
    endpoint: String,
    topic: String,
    dp_rank: u32,
    cache_state_api_token: Option<String>,
    delivery_config: DeliveryConfig,
    max_sink_payload_bytes: usize,
    metrics: Arc<AgentMetrics>,
    cancel: CancellationToken,
}

impl AgentTask {
    async fn run(self) {
        let worker = KvWorkerId::new(self.worker_url.clone(), self.dp_rank);
        let mut dispatchers = Vec::with_capacity(self.sinks.len());
        let mut sink_handles = Vec::with_capacity(self.sinks.len());
        for sink in self.sinks.iter().cloned() {
            let (dispatcher, handle) = spawn_sink_worker(
                sink,
                self.client.clone(),
                self.cache_state_api_token.clone(),
                self.delivery_config,
                Arc::clone(&self.metrics),
                self.cancel.clone(),
            );
            dispatchers.push(dispatcher);
            sink_handles.push(handle);
        }
        let mut cursor = PublisherCursor::default();
        loop {
            if self.cancel.is_cancelled() {
                break;
            }
            match self.connect().await {
                Some(mut sub) => {
                    self.recv_loop(&worker, &mut sub, &dispatchers, &mut cursor)
                        .await
                }
                None => break,
            }
            tokio::select! {
                _ = self.cancel.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(2)) => {}
            }
        }
        drop(dispatchers);
        for handle in sink_handles {
            let _ = handle.await;
        }
    }

    async fn connect(&self) -> Option<SubSocket> {
        loop {
            let mut sub = SubSocket::new();
            info!(
                endpoint = %self.endpoint,
                dp_rank = self.dp_rank,
                "connecting to local KV event publisher"
            );
            let connect_result = tokio::select! {
                _ = self.cancel.cancelled() => return None,
                res = sub.connect(&self.endpoint) => res,
            };
            if let Err(err) = connect_result {
                warn!(
                    endpoint = %self.endpoint,
                    dp_rank = self.dp_rank,
                    error = %err,
                    "connect failed; retrying"
                );
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
            let subscribe_result = tokio::select! {
                _ = self.cancel.cancelled() => return None,
                res = sub.subscribe(&self.topic) => res,
            };
            match subscribe_result {
                Ok(()) => return Some(sub),
                Err(err) => {
                    warn!(
                        endpoint = %self.endpoint,
                        dp_rank = self.dp_rank,
                        error = %err,
                        "subscribe failed; retrying"
                    );
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    async fn recv_loop(
        &self,
        worker: &KvWorkerId,
        sub: &mut SubSocket,
        dispatchers: &[SinkDispatcher],
        cursor: &mut PublisherCursor,
    ) {
        loop {
            let msg = tokio::select! {
                _ = self.cancel.cancelled() => return,
                res = sub.recv() => match res {
                    Ok(msg) => msg,
                    Err(err) => {
                        warn!(
                            dp_rank = self.dp_rank,
                            error = %err,
                            "recv failed; reconnecting"
                        );
                        return;
                    }
                },
            };
            let Some((seq, payload)) = decode_zmq_message(&msg, self.dp_rank) else {
                continue;
            };
            if seq == END_SEQ_SENTINEL {
                info!(
                    dp_rank = self.dp_rank,
                    "publisher shutdown sentinel received"
                );
                continue;
            }
            if payload.len() > self.max_sink_payload_bytes {
                self.metrics
                    .oversized_payloads
                    .fetch_add(1, Ordering::Relaxed);
                warn!(
                    dp_rank = self.dp_rank,
                    seq,
                    payload_bytes = payload.len(),
                    max_sink_payload_bytes = self.max_sink_payload_bytes,
                    "dropping KV event payload that exceeds the configured sink limit"
                );
                continue;
            }
            let batch = match decode_event_batch(payload) {
                Ok(batch) => batch,
                Err(err) => {
                    warn!(
                        dp_rank = self.dp_rank,
                        seq,
                        error = %err,
                        "failed to decode KV event batch"
                    );
                    continue;
                }
            };
            cursor.observe(
                batch.publisher_epoch.as_deref(),
                seq,
                self.dp_rank,
                &self.metrics,
            );
            let n_events = batch.events.len();
            let record = Arc::new(KvEventStreamRecord::from_payload(
                self.model_id.clone(),
                worker.url.clone(),
                worker.dp_rank,
                seq,
                payload,
            ));
            let delivery = SinkDelivery { record, n_events };
            for dispatcher in dispatchers {
                dispatcher.try_send(delivery.clone());
            }
        }
    }
}

#[derive(Debug, Default)]
struct PublisherCursor {
    initialized: bool,
    epoch: Option<String>,
    last_seq: i64,
}

impl PublisherCursor {
    fn observe(&mut self, epoch: Option<&str>, seq: i64, dp_rank: u32, metrics: &AgentMetrics) {
        if !self.initialized || self.epoch.as_deref() != epoch {
            self.initialized = true;
            self.epoch = epoch.map(str::to_owned);
            self.last_seq = seq;
            return;
        }
        if seq > self.last_seq.saturating_add(1) {
            metrics
                .publisher_sequence_gaps
                .fetch_add(1, Ordering::Relaxed);
            warn!(
                dp_rank,
                publisher_epoch = epoch.unwrap_or("legacy"),
                expected_seq = self.last_seq.saturating_add(1),
                observed_seq = seq,
                "detected publisher sequence gap"
            );
        }
        if epoch.is_none() && seq <= self.last_seq {
            // Legacy payloads have no generation identity. A sequence reset
            // is the only observable restart signal.
            self.last_seq = seq;
        } else if seq > self.last_seq {
            self.last_seq = seq;
        }
    }
}

fn decode_zmq_message(msg: &zeromq::ZmqMessage, dp_rank: u32) -> Option<(i64, &[u8])> {
    if msg.len() != 3 {
        warn!(
            dp_rank,
            frames = msg.len(),
            "dropping ZMQ message with unexpected frame count"
        );
        return None;
    }
    let seq_frame = msg.get(1)?;
    let payload = msg.get(2)?;
    let seq_bytes: [u8; 8] = match seq_frame.as_ref().try_into() {
        Ok(bytes) => bytes,
        Err(_) => {
            warn!(
                dp_rank,
                seq_len = seq_frame.len(),
                "dropping message with invalid seq frame"
            );
            return None;
        }
    };
    Some((i64::from_be_bytes(seq_bytes), payload.as_ref()))
}

fn install_signal_handlers(cancel: CancellationToken) -> Result<()> {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("install SIGTERM handler")?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .context("install SIGINT handler")?;
    tokio::spawn(async move {
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }
        cancel.cancel();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_delivery(seq: i64) -> SinkDelivery {
        SinkDelivery {
            record: Arc::new(KvEventStreamRecord::from_payload(
                "m".into(),
                "http://worker:30000".into(),
                0,
                seq,
                b"payload",
            )),
            n_events: 1,
        }
    }

    fn delivery_config(queue_capacity: usize) -> DeliveryConfig {
        DeliveryConfig {
            queue_capacity,
            max_attempts: 1,
            retry_backoff: Duration::from_millis(1),
            delivery_timeout: Duration::from_secs(2),
        }
    }

    #[tokio::test]
    async fn slow_sink_does_not_block_fast_sink() {
        let metrics = Arc::new(AgentMetrics::default());
        let cancel = CancellationToken::new();
        let slow_delivered = Arc::new(AtomicU64::new(0));
        let fast_delivered = Arc::new(AtomicU64::new(0));
        let client = Client::new();
        let (slow, slow_handle) = spawn_sink_worker(
            EventSink::Test {
                name: "slow",
                kind: SinkKind::Kafka,
                delay: Duration::from_millis(200),
                delivered: Arc::clone(&slow_delivered),
            },
            client.clone(),
            None,
            delivery_config(4),
            Arc::clone(&metrics),
            cancel.clone(),
        );
        let (fast, fast_handle) = spawn_sink_worker(
            EventSink::Test {
                name: "fast",
                kind: SinkKind::Http,
                delay: Duration::ZERO,
                delivered: Arc::clone(&fast_delivered),
            },
            client,
            None,
            delivery_config(4),
            Arc::clone(&metrics),
            cancel.clone(),
        );

        let delivery = test_delivery(1);
        slow.try_send(delivery.clone());
        fast.try_send(delivery);
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(fast_delivered.load(Ordering::Relaxed), 1);
        assert_eq!(slow_delivered.load(Ordering::Relaxed), 0);
        tokio::time::sleep(Duration::from_millis(220)).await;
        assert_eq!(slow_delivered.load(Ordering::Relaxed), 1);

        cancel.cancel();
        let _ = slow_handle.await;
        let _ = fast_handle.await;
    }

    #[tokio::test]
    async fn full_sink_queue_records_bounded_drop_metric() {
        let metrics = Arc::new(AgentMetrics::default());
        let cancel = CancellationToken::new();
        let delivered = Arc::new(AtomicU64::new(0));
        let (dispatcher, handle) = spawn_sink_worker(
            EventSink::Test {
                name: "slow",
                kind: SinkKind::Kafka,
                delay: Duration::from_secs(1),
                delivered,
            },
            Client::new(),
            None,
            delivery_config(1),
            Arc::clone(&metrics),
            cancel.clone(),
        );
        for seq in 0..20 {
            dispatcher.try_send(test_delivery(seq));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(metrics.queue_drops_kafka.load(Ordering::Relaxed) > 0);
        let rendered = metrics.render();
        assert!(rendered
            .contains("sgl_router_cache_event_agent_sink_queue_drops_total{sink=\"kafka\"}"));
        assert!(!rendered.contains("http://worker:30000"));

        cancel.cancel();
        let _ = handle.await;
    }

    #[test]
    fn publisher_cursor_counts_only_same_epoch_forward_gaps() {
        let metrics = AgentMetrics::default();
        let mut cursor = PublisherCursor::default();
        cursor.observe(Some("epoch-a"), 0, 0, &metrics);
        cursor.observe(Some("epoch-a"), 2, 0, &metrics);
        cursor.observe(Some("epoch-b"), 0, 0, &metrics);
        cursor.observe(Some("epoch-b"), 1, 0, &metrics);
        assert_eq!(metrics.publisher_sequence_gaps.load(Ordering::Relaxed), 1);
    }
}
