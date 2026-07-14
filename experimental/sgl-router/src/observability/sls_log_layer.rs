//! Alibaba Cloud SLS direct-push logging for the production router.
//!
//! The tracing callback only serializes an event and attempts a non-blocking
//! enqueue into a bounded channel. A dedicated OS thread owns all network I/O,
//! batches up to 200 events, and flushes at least once per second.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use chrono::Utc;
use hmac::{Hmac, Mac};
use md5::{Digest, Md5};
use reqwest::{
    blocking::Client as BlockingClient,
    header::{HeaderValue, AUTHORIZATION},
};
use sha1::Sha1;
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Event, Subscriber,
};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

use crate::server::metrics::{MetricsRegistry, SlsBatchResult, SlsDropReason};

const SLS_BATCH_SIZE: usize = 200;
const SLS_QUEUE_CAPACITY: usize = 10_000;
const SLS_FLUSH_INTERVAL: Duration = Duration::from_secs(1);
const SLS_HTTP_TIMEOUT: Duration = Duration::from_secs(5);
// Keep process termination bounded even if a full queue accumulated.
const SLS_SHUTDOWN_BATCH_LIMIT: usize = 2;

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type HmacSha1 = Hmac<Sha1>;

#[derive(Debug, thiserror::Error)]
pub enum SlsConfigError {
    #[error("SLS logging is partially configured; missing {0}")]
    MissingVariables(String),
}

/// Configuration for direct SLS writes. Secret values are always redacted.
#[derive(Clone)]
pub struct SlsLayerConfig {
    pub endpoint: String,
    pub access_key_id: String,
    pub access_key_secret: String,
    pub project: String,
    pub logstore: String,
    pub service_name: String,
}

impl std::fmt::Debug for SlsLayerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlsLayerConfig")
            .field("endpoint", &self.endpoint)
            .field("access_key_id", &"<redacted>")
            .field("access_key_secret", &"<redacted>")
            .field("project", &self.project)
            .field("logstore", &self.logstore)
            .field("service_name", &self.service_name)
            .finish()
    }
}

impl SlsLayerConfig {
    /// Read configuration from the process environment.
    ///
    /// No SLS variables means disabled. Partial credentials are an explicit
    /// configuration error so a rollout cannot silently lose remote logs.
    pub fn from_env(service_name: &str) -> Result<Option<Self>, SlsConfigError> {
        Self::from_lookup(service_name, |name| std::env::var(name).ok())
    }

    fn from_lookup<F>(service_name: &str, lookup: F) -> Result<Option<Self>, SlsConfigError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let clean = |name: &str| {
            lookup(name)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        let endpoint = clean("SLS_ENDPOINT");
        let access_key_id = clean("SLS_ACCESS_KEY_ID");
        let access_key_secret = clean("SLS_ACCESS_KEY_SECRET");

        if endpoint.is_none() && access_key_id.is_none() && access_key_secret.is_none() {
            return Ok(None);
        }

        let missing = [
            ("SLS_ENDPOINT", endpoint.is_none()),
            ("SLS_ACCESS_KEY_ID", access_key_id.is_none()),
            ("SLS_ACCESS_KEY_SECRET", access_key_secret.is_none()),
        ]
        .into_iter()
        .filter_map(|(name, absent)| absent.then_some(name))
        .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(SlsConfigError::MissingVariables(missing.join(", ")));
        }

        let endpoint = endpoint
            .expect("checked above")
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string();
        let project = clean("SLS_PROJECT").unwrap_or_else(|| "macaron-log".to_string());
        let logstore = clean("SLS_LOGSTORE").unwrap_or_else(|| "sglang-router".to_string());

        Ok(Some(Self {
            endpoint,
            access_key_id: access_key_id.expect("checked above"),
            access_key_secret: access_key_secret.expect("checked above"),
            project,
            logstore,
            service_name: service_name.to_string(),
        }))
    }
}

#[derive(Debug)]
struct SlsLogEntry {
    timestamp: u32,
    contents: Vec<(String, String)>,
}

enum SlsCommand {
    Entry(SlsLogEntry),
    Shutdown,
}

trait SlsBatchSink: Send + 'static {
    fn send_batch(&mut self, entries: Vec<SlsLogEntry>) -> Result<(), BoxError>;
}

struct SlsHttpSink {
    config: SlsLayerConfig,
    client: BlockingClient,
}

impl SlsHttpSink {
    fn new(config: SlsLayerConfig) -> Result<Self, reqwest::Error> {
        let client = BlockingClient::builder()
            .timeout(SLS_HTTP_TIMEOUT)
            .build()?;
        Ok(Self { config, client })
    }

    fn build_request(
        &self,
        entries: Vec<SlsLogEntry>,
    ) -> Result<reqwest::blocking::Request, BoxError> {
        let body = encode_log_group_pb(&entries);
        let content_md5 = format!("{:X}", Md5::digest(&body));
        let date = Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string();
        let resource = format!("/logstores/{}/shards/lb", self.config.logstore);
        let content_type = "application/x-protobuf";

        let mut sign_headers = vec![
            ("x-log-apiversion", "0.6.0".to_string()),
            ("x-log-bodyrawsize", body.len().to_string()),
            ("x-log-signaturemethod", "hmac-sha1".to_string()),
        ];
        sign_headers.sort_by(|a, b| a.0.cmp(b.0));
        let canonical_log_headers: String = sign_headers
            .iter()
            .map(|(key, value)| format!("{key}:{value}\n"))
            .collect();
        let sign_content = format!(
            "POST\n{}\n{}\n{}\n{}{}",
            content_md5, content_type, date, canonical_log_headers, resource
        );

        let mut mac = HmacSha1::new_from_slice(self.config.access_key_secret.as_bytes())?;
        mac.update(sign_content.as_bytes());
        let signature =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());
        let mut authorization =
            HeaderValue::from_str(&format!("LOG {}:{}", self.config.access_key_id, signature))?;
        authorization.set_sensitive(true);

        let url = format!(
            "https://{}.{}{}",
            self.config.project, self.config.endpoint, resource
        );
        let mut request = self
            .client
            .post(url)
            .header("Content-Type", content_type)
            .header("Content-MD5", content_md5)
            .header("Date", &date)
            // SLS AuthV1 adds x-log-date after calculating the signature.
            .header("x-log-date", &date)
            .header(AUTHORIZATION, authorization)
            .body(body);
        for (key, value) in sign_headers {
            request = request.header(key, value);
        }
        Ok(request.build()?)
    }
}

impl SlsBatchSink for SlsHttpSink {
    fn send_batch(&mut self, entries: Vec<SlsLogEntry>) -> Result<(), BoxError> {
        let response = self.client.execute(self.build_request(entries)?)?;
        if !response.status().is_success() {
            return Err(format!("SLS PutLogs returned {}", response.status()).into());
        }
        Ok(())
    }
}

/// Stops the worker thread and performs a bounded final flush on drop.
pub struct SlsWorkerGuard {
    accepting: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shutdown_tx: Option<SyncSender<SlsCommand>>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for SlsWorkerGuard {
    fn drop(&mut self) {
        self.accepting.store(false, Ordering::Release);
        self.stop.store(true, Ordering::Release);
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            // The queue may be full. The stop flag independently wakes the
            // loop after its next receive, so shutdown never blocks here.
            let _ = shutdown_tx.try_send(SlsCommand::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                eprintln!("[sls-log-layer] worker thread panicked");
            }
        }
    }
}

fn spawn_sls_worker<S: SlsBatchSink>(
    sink: S,
    metrics: Arc<MetricsRegistry>,
    queue_capacity: usize,
    batch_size: usize,
    flush_interval: Duration,
) -> Result<(SyncSender<SlsCommand>, SlsWorkerGuard), std::io::Error> {
    let (tx, rx) = mpsc::sync_channel(queue_capacity);
    let accepting = Arc::new(AtomicBool::new(true));
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let worker = thread::Builder::new()
        .name("sls-log-pusher".to_string())
        .spawn(move || {
            run_sls_worker(rx, sink, metrics, worker_stop, batch_size, flush_interval)
        })?;

    Ok((
        tx.clone(),
        SlsWorkerGuard {
            accepting,
            stop,
            shutdown_tx: Some(tx),
            worker: Some(worker),
        },
    ))
}

fn run_sls_worker<S: SlsBatchSink>(
    rx: Receiver<SlsCommand>,
    mut sink: S,
    metrics: Arc<MetricsRegistry>,
    stop: Arc<AtomicBool>,
    batch_size: usize,
    flush_interval: Duration,
) {
    let mut batch = Vec::with_capacity(batch_size);
    let mut flush_deadline = Instant::now() + flush_interval;

    loop {
        if stop.load(Ordering::Acquire) {
            drain_and_flush(&rx, &mut sink, &metrics, &mut batch, batch_size);
            return;
        }

        let timeout = flush_deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(timeout) {
            Ok(SlsCommand::Entry(entry)) => {
                metrics.adjust_sls_queue_depth(-1);
                batch.push(entry);
                if batch.len() >= batch_size {
                    flush_batch(&mut sink, &metrics, &mut batch, batch_size);
                    flush_deadline = Instant::now() + flush_interval;
                }
            }
            Ok(SlsCommand::Shutdown) => {
                drain_and_flush(&rx, &mut sink, &metrics, &mut batch, batch_size);
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                flush_batch(&mut sink, &metrics, &mut batch, batch_size);
                flush_deadline = Instant::now() + flush_interval;
            }
            Err(RecvTimeoutError::Disconnected) => {
                flush_batch(&mut sink, &metrics, &mut batch, batch_size);
                return;
            }
        }
    }
}

fn drain_and_flush<S: SlsBatchSink>(
    rx: &Receiver<SlsCommand>,
    sink: &mut S,
    metrics: &MetricsRegistry,
    batch: &mut Vec<SlsLogEntry>,
    batch_size: usize,
) {
    let mut flushed_batches = 0;
    loop {
        while batch.len() < batch_size {
            match rx.try_recv() {
                Ok(SlsCommand::Entry(entry)) => {
                    metrics.adjust_sls_queue_depth(-1);
                    batch.push(entry);
                }
                Ok(SlsCommand::Shutdown) => {}
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }

        if batch.is_empty() {
            return;
        }
        if flushed_batches >= SLS_SHUTDOWN_BATCH_LIMIT {
            let mut dropped = batch.len() as u64;
            batch.clear();
            loop {
                match rx.try_recv() {
                    Ok(SlsCommand::Entry(_)) => {
                        metrics.adjust_sls_queue_depth(-1);
                        dropped += 1;
                    }
                    Ok(SlsCommand::Shutdown) => {}
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                }
            }
            metrics.record_sls_drop(SlsDropReason::ShutdownLimit, dropped);
            return;
        }

        flush_batch(sink, metrics, batch, batch_size);
        flushed_batches += 1;
    }
}

fn flush_batch<S: SlsBatchSink>(
    sink: &mut S,
    metrics: &MetricsRegistry,
    batch: &mut Vec<SlsLogEntry>,
    batch_size: usize,
) {
    if batch.is_empty() {
        return;
    }

    let entry_count = batch.len() as u64;
    let entries = std::mem::replace(batch, Vec::with_capacity(batch_size));
    match sink.send_batch(entries) {
        Ok(()) => {
            metrics.record_sls_batch(SlsBatchResult::Success);
            metrics.record_sls_entries_sent(entry_count);
        }
        Err(error) => {
            metrics.record_sls_batch(SlsBatchResult::Error);
            metrics.record_sls_drop(SlsDropReason::SendError, entry_count);
            eprintln!("[sls-log-layer] flush error: {error}");
        }
    }
}

#[derive(Clone, Default)]
struct RecordedFields(BTreeMap<String, String>);

#[derive(Default)]
struct FieldVisitor {
    fields: BTreeMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }
}

/// A tracing layer whose callback performs no network I/O.
pub struct SlsLogLayer {
    tx: SyncSender<SlsCommand>,
    accepting: Arc<AtomicBool>,
    metrics: Arc<MetricsRegistry>,
    service_name: String,
}

impl SlsLogLayer {
    pub fn new(
        config: SlsLayerConfig,
        metrics: Arc<MetricsRegistry>,
    ) -> Result<(Self, SlsWorkerGuard), BoxError> {
        let service_name = config.service_name.clone();
        let sink = SlsHttpSink::new(config)?;
        let (tx, guard) = spawn_sls_worker(
            sink,
            Arc::clone(&metrics),
            SLS_QUEUE_CAPACITY,
            SLS_BATCH_SIZE,
            SLS_FLUSH_INTERVAL,
        )?;
        let accepting = Arc::clone(&guard.accepting);
        Ok((
            Self {
                tx,
                accepting,
                metrics,
                service_name,
            },
            guard,
        ))
    }

    fn enqueue(&self, entry: SlsLogEntry) {
        if !self.accepting.load(Ordering::Acquire) {
            self.metrics
                .record_sls_drop(SlsDropReason::WorkerStopped, 1);
            return;
        }

        self.metrics.adjust_sls_queue_depth(1);
        match self.tx.try_send(SlsCommand::Entry(entry)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.metrics.adjust_sls_queue_depth(-1);
                self.metrics.record_sls_drop(SlsDropReason::QueueFull, 1);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.metrics.adjust_sls_queue_depth(-1);
                self.metrics
                    .record_sls_drop(SlsDropReason::WorkerStopped, 1);
            }
        }
    }
}

impl<S> Layer<S> for SlsLogLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(RecordedFields(visitor.fields));
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        let mut extensions = span.extensions_mut();
        if let Some(fields) = extensions.get_mut::<RecordedFields>() {
            fields.0.extend(visitor.fields);
        } else {
            extensions.insert(RecordedFields(visitor.fields));
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut fields = BTreeMap::new();
        let mut span_name = None;
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                span_name = Some(span.metadata().name().to_string());
                if let Some(recorded) = span.extensions().get::<RecordedFields>() {
                    fields.extend(recorded.0.clone());
                }
            }
        }

        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        fields.extend(visitor.fields);
        let message = fields
            .remove("message")
            .unwrap_or_else(|| metadata.name().to_string());
        if let Some(span_name) = span_name {
            fields.entry("span".to_string()).or_insert(span_name);
        }

        let unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let timestamp = u32::try_from(unix_seconds).unwrap_or(u32::MAX);
        let mut contents = Vec::with_capacity(fields.len() + 8);
        contents.push((
            "level".to_string(),
            metadata.level().to_string().to_lowercase(),
        ));
        contents.push(("target".to_string(), metadata.target().to_string()));
        contents.push(("event_name".to_string(), metadata.name().to_string()));
        contents.push(("message".to_string(), message));
        contents.push(("service_name".to_string(), self.service_name.clone()));
        contents.push(("timestamp".to_string(), Utc::now().to_rfc3339()));
        if let Some(module_path) = metadata.module_path() {
            contents.push(("module_path".to_string(), module_path.to_string()));
        }
        if let Some(line) = metadata.line() {
            contents.push(("line".to_string(), line.to_string()));
        }
        contents.extend(fields);

        self.enqueue(SlsLogEntry {
            timestamp,
            contents,
        });
    }
}

// Minimal protobuf wire encoding for LogGroup { logs: [Log] }.

fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    while value >= 0x80 {
        result.push((value as u8) | 0x80);
        value >>= 7;
    }
    result.push(value as u8);
    result
}

fn encode_tag(field_number: u32, wire_type: u32) -> Vec<u8> {
    encode_varint(((field_number as u64) << 3) | wire_type as u64)
}

fn encode_string(field_number: u32, value: &str) -> Vec<u8> {
    let mut result = encode_tag(field_number, 2);
    result.extend(encode_varint(value.len() as u64));
    result.extend_from_slice(value.as_bytes());
    result
}

fn encode_uint32(field_number: u32, value: u32) -> Vec<u8> {
    let mut result = encode_tag(field_number, 0);
    result.extend(encode_varint(value as u64));
    result
}

fn encode_content(key: &str, value: &str) -> Vec<u8> {
    let mut result = Vec::new();
    result.extend(encode_string(1, key));
    result.extend(encode_string(2, value));
    result
}

fn encode_log(entry: &SlsLogEntry) -> Vec<u8> {
    let mut result = Vec::new();
    result.extend(encode_uint32(1, entry.timestamp));
    for (key, value) in &entry.contents {
        let content = encode_content(key, value);
        result.extend(encode_tag(2, 2));
        result.extend(encode_varint(content.len() as u64));
        result.extend(content);
    }
    result
}

fn encode_log_group_pb(entries: &[SlsLogEntry]) -> Vec<u8> {
    let mut result = Vec::new();
    for entry in entries {
        let log = encode_log(entry);
        result.extend(encode_tag(1, 2));
        result.extend(encode_varint(log.len() as u64));
        result.extend(log);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    struct RecordingSink {
        batches: mpsc::Sender<Vec<String>>,
    }

    impl SlsBatchSink for RecordingSink {
        fn send_batch(&mut self, entries: Vec<SlsLogEntry>) -> Result<(), BoxError> {
            let messages = entries
                .into_iter()
                .map(|entry| {
                    entry
                        .contents
                        .into_iter()
                        .find_map(|(key, value)| (key == "message").then_some(value))
                        .unwrap_or_default()
                })
                .collect();
            self.batches.send(messages)?;
            Ok(())
        }
    }

    fn test_entry(message: &str) -> SlsLogEntry {
        SlsLogEntry {
            timestamp: 1,
            contents: vec![("message".to_string(), message.to_string())],
        }
    }

    fn test_layer<S: SlsBatchSink>(
        sink: S,
        metrics: Arc<MetricsRegistry>,
        queue_capacity: usize,
        batch_size: usize,
        flush_interval: Duration,
    ) -> (SlsLogLayer, SlsWorkerGuard) {
        let (tx, guard) = spawn_sls_worker(
            sink,
            Arc::clone(&metrics),
            queue_capacity,
            batch_size,
            flush_interval,
        )
        .unwrap();
        (
            SlsLogLayer {
                tx,
                accepting: Arc::clone(&guard.accepting),
                metrics,
                service_name: "sglang-router".to_string(),
            },
            guard,
        )
    }

    #[test]
    fn worker_flushes_at_batch_size() {
        let (batch_tx, batch_rx) = mpsc::channel();
        let metrics = MetricsRegistry::new();
        let (layer, guard) = test_layer(
            RecordingSink { batches: batch_tx },
            metrics,
            8,
            2,
            Duration::from_secs(30),
        );

        layer.enqueue(test_entry("one"));
        layer.enqueue(test_entry("two"));
        assert_eq!(
            batch_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec!["one", "two"]
        );
        drop(guard);
    }

    #[test]
    fn worker_flushes_on_fixed_interval() {
        let (batch_tx, batch_rx) = mpsc::channel();
        let metrics = MetricsRegistry::new();
        let (layer, guard) = test_layer(
            RecordingSink { batches: batch_tx },
            metrics,
            8,
            10,
            Duration::from_millis(20),
        );

        layer.enqueue(test_entry("one"));
        assert_eq!(
            batch_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec!["one"]
        );
        drop(guard);
    }

    #[test]
    fn worker_drains_on_shutdown() {
        let (batch_tx, batch_rx) = mpsc::channel();
        let metrics = MetricsRegistry::new();
        let (layer, guard) = test_layer(
            RecordingSink { batches: batch_tx },
            metrics,
            8,
            10,
            Duration::from_secs(30),
        );

        layer.enqueue(test_entry("one"));
        layer.enqueue(test_entry("two"));
        drop(guard);
        assert_eq!(
            batch_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec!["one", "two"]
        );
    }

    #[test]
    fn layer_drops_when_queue_is_full() {
        let (tx, rx) = mpsc::sync_channel(1);
        let metrics = MetricsRegistry::new();
        let layer = SlsLogLayer {
            tx,
            accepting: Arc::new(AtomicBool::new(true)),
            metrics: Arc::clone(&metrics),
            service_name: "sglang-router".to_string(),
        };

        layer.enqueue(test_entry("queued"));
        layer.enqueue(test_entry("dropped"));
        let SlsCommand::Entry(entry) = rx.try_recv().unwrap() else {
            panic!("expected a log entry");
        };
        assert_eq!(entry.contents[0].1, "queued");
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
        assert!(metrics
            .render()
            .contains("sgl_router_sls_log_entries_dropped_total{reason=\"queue_full\"} 1"));
    }

    #[test]
    fn layer_preserves_event_and_request_span_fields() {
        let (tx, rx) = mpsc::sync_channel(1);
        let layer = SlsLogLayer {
            tx,
            accepting: Arc::new(AtomicBool::new(true)),
            metrics: MetricsRegistry::new(),
            service_name: "sglang-router".to_string(),
        };
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "http_request",
                trace_id = "trace-1",
                request_id = "request-1"
            );
            let _entered = span.enter();
            tracing::info!(worker = "worker-1", "hello {}", "world");
        });

        let SlsCommand::Entry(entry) = rx.recv_timeout(Duration::from_secs(1)).unwrap() else {
            panic!("expected a log entry");
        };
        let fields: BTreeMap<_, _> = entry.contents.into_iter().collect();
        assert_eq!(fields.get("message").unwrap(), "hello world");
        assert_eq!(fields.get("trace_id").unwrap(), "trace-1");
        assert_eq!(fields.get("request_id").unwrap(), "request-1");
        assert_eq!(fields.get("worker").unwrap(), "worker-1");
    }

    #[test]
    fn request_has_one_copy_of_each_signed_header_and_redacts_auth() {
        let sink = SlsHttpSink::new(SlsLayerConfig {
            endpoint: "example.log.aliyuncs.com".to_string(),
            access_key_id: "fake_access_key".to_string(),
            access_key_secret: "fake_secret".to_string(),
            project: "test-project".to_string(),
            logstore: "test-logstore".to_string(),
            service_name: "sglang-router".to_string(),
        })
        .unwrap();

        let request = sink.build_request(vec![test_entry("hello")]).unwrap();
        for header_name in [
            "x-log-apiversion",
            "x-log-bodyrawsize",
            "x-log-signaturemethod",
        ] {
            assert_eq!(request.headers().get_all(header_name).iter().count(), 1);
        }
        assert!(request.headers()[AUTHORIZATION].is_sensitive());
        assert_eq!(
            request.url().as_str(),
            "https://test-project.example.log.aliyuncs.com/logstores/test-logstore/shards/lb"
        );
    }

    #[test]
    fn protobuf_helpers_encode_expected_wire_bytes() {
        assert_eq!(encode_varint(300), vec![0xac, 0x02]);
        assert_eq!(
            encode_string(1, "hello"),
            vec![0x0a, 0x05, b'h', b'e', b'l', b'l', b'o']
        );
        let encoded = encode_log_group_pb(&[test_entry("hello")]);
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0], 0x0a);
    }

    #[test]
    fn config_lookup_detects_partial_values_and_redacts_secrets() {
        let values = BTreeMap::from([
            ("SLS_ENDPOINT", "https://example.log.aliyuncs.com"),
            ("SLS_ACCESS_KEY_ID", "fake_access_key"),
            ("SLS_ACCESS_KEY_SECRET", "fake_secret"),
            ("SLS_PROJECT", "test-project"),
        ]);
        let config = SlsLayerConfig::from_lookup("sglang-router", |name| {
            values.get(name).map(|value| value.to_string())
        })
        .unwrap()
        .unwrap();
        assert_eq!(config.endpoint, "example.log.aliyuncs.com");
        assert_eq!(config.logstore, "sglang-router");
        let debug = format!("{config:?}");
        assert!(!debug.contains("fake_access_key"));
        assert!(!debug.contains("fake_secret"));

        let error = SlsLayerConfig::from_lookup("sglang-router", |name| {
            (name == "SLS_ENDPOINT").then(|| "example.log.aliyuncs.com".to_string())
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("SLS_ACCESS_KEY_ID"));
        assert!(error.contains("SLS_ACCESS_KEY_SECRET"));
    }
}
