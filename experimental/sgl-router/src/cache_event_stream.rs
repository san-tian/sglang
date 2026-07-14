// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bounded KV event stream primitives for cache-state fanout.
//!
//! The local append-log backend is intentionally simple: it gives tests and
//! non-production deployments a durable stream shape without requiring cloud
//! infrastructure. Production multi-replica cache-state should use the same
//! envelope with a managed fanout backend.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use rdkafka::client::ClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, ConsumerContext, StreamConsumer};
use rdkafka::error::KafkaResult;
use rdkafka::message::BorrowedMessage;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::{Message, Offset, TopicPartitionList};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_KAFKA_AUTO_COMMIT_INTERVAL_MS: u32 = 1_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvEventStreamRecord {
    pub schema_version: u32,
    pub worker_url: String,
    pub model_id: String,
    #[serde(default)]
    pub dp_rank: u32,
    pub seq: i64,
    pub observed_at_ms: u64,
    pub payload_hash: String,
    pub payload_b64: String,
}

impl KvEventStreamRecord {
    pub fn from_payload(
        model_id: String,
        worker_url: String,
        dp_rank: u32,
        seq: i64,
        payload: &[u8],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        Self {
            schema_version: SCHEMA_VERSION,
            worker_url,
            model_id,
            dp_rank,
            seq,
            observed_at_ms: now_ms(),
            payload_hash: format!("{:x}", hasher.finalize()),
            payload_b64: encode_base64(payload),
        }
    }

    pub fn dedupe_key(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}",
            self.worker_url, self.dp_rank, self.seq, self.observed_at_ms, self.payload_hash
        )
    }
}

#[derive(Debug, Clone)]
pub struct LocalKvEventStreamConfig {
    pub path: PathBuf,
    pub retention_secs: Option<u64>,
    pub max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendStats {
    pub records_after_compaction: usize,
    pub bytes_after_compaction: u64,
}

#[derive(Debug, Clone)]
pub struct LocalKvEventStream {
    config: Arc<LocalKvEventStreamConfig>,
    lock: Arc<Mutex<()>>,
}

#[derive(Debug, Clone)]
pub struct KafkaKvEventStreamConfig {
    pub bootstrap_servers: String,
    pub topic: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub client_id: Option<String>,
    pub consumer_group: Option<String>,
    pub auto_offset_reset: String,
    pub auto_commit_interval_ms: u32,
}

impl KafkaKvEventStreamConfig {
    pub fn producer_from_env(prefix: &str) -> Result<Option<Self>> {
        let Some(bootstrap_servers) = env_string(&format!("{prefix}_KAFKA_BOOTSTRAP_SERVERS"))
        else {
            return Ok(None);
        };
        let topic = env_string(&format!("{prefix}_KAFKA_TOPIC"))
            .ok_or_else(|| anyhow!("{prefix}_KAFKA_TOPIC is required when Kafka is enabled"))?;
        Ok(Some(Self {
            bootstrap_servers,
            topic,
            username: env_string(&format!("{prefix}_KAFKA_USERNAME")),
            password: env_string(&format!("{prefix}_KAFKA_PASSWORD")),
            client_id: env_string(&format!("{prefix}_KAFKA_CLIENT_ID")),
            consumer_group: None,
            auto_offset_reset: "latest".to_string(),
            auto_commit_interval_ms: DEFAULT_KAFKA_AUTO_COMMIT_INTERVAL_MS,
        }))
    }

    pub fn consumer_from_env(prefix: &str) -> Result<Option<Self>> {
        let Some(bootstrap_servers) = env_string(&format!("{prefix}_KAFKA_BOOTSTRAP_SERVERS"))
        else {
            return Ok(None);
        };
        let topic = env_string(&format!("{prefix}_KAFKA_TOPIC"))
            .ok_or_else(|| anyhow!("{prefix}_KAFKA_TOPIC is required when Kafka is enabled"))?;
        let consumer_group =
            env_string(&format!("{prefix}_KAFKA_CONSUMER_GROUP")).ok_or_else(|| {
                anyhow!("{prefix}_KAFKA_CONSUMER_GROUP is required for cache-state Kafka consumer")
            })?;
        let auto_commit_interval_ms =
            parse_auto_commit_interval_ms(prefix, DEFAULT_KAFKA_AUTO_COMMIT_INTERVAL_MS)?;
        Ok(Some(Self {
            bootstrap_servers,
            topic,
            username: env_string(&format!("{prefix}_KAFKA_USERNAME")),
            password: env_string(&format!("{prefix}_KAFKA_PASSWORD")),
            client_id: env_string(&format!("{prefix}_KAFKA_CLIENT_ID")),
            consumer_group: Some(consumer_group),
            auto_offset_reset: env_string(&format!("{prefix}_KAFKA_AUTO_OFFSET_RESET"))
                .unwrap_or_else(|| "earliest".to_string()),
            auto_commit_interval_ms,
        }))
    }

    fn base_client_config(&self) -> ClientConfig {
        let mut cfg = ClientConfig::new();
        cfg.set("bootstrap.servers", &self.bootstrap_servers);
        cfg.set("message.timeout.ms", "5000");
        cfg.set("socket.keepalive.enable", "true");
        cfg.set("security.protocol", "SASL_SSL");
        cfg.set("sasl.mechanism", "PLAIN");
        if let Some(username) = self.username.as_ref() {
            cfg.set("sasl.username", username);
        }
        if let Some(password) = self.password.as_ref() {
            cfg.set("sasl.password", password);
        }
        if let Some(client_id) = self.client_id.as_ref() {
            cfg.set("client.id", client_id);
        }
        cfg
    }

    fn consumer_client_config(&self, group: &str) -> ClientConfig {
        let mut cfg = self.base_client_config();
        cfg.set("group.id", group)
            .set("enable.auto.commit", "true")
            .set(
                "auto.commit.interval.ms",
                self.auto_commit_interval_ms.to_string(),
            )
            .set("enable.auto.offset.store", "false")
            .set("auto.offset.reset", &self.auto_offset_reset)
            .set("session.timeout.ms", "30000");
        cfg
    }
}

fn parse_auto_commit_interval_ms(prefix: &str, default: u32) -> Result<u32> {
    let name = format!("{prefix}_KAFKA_AUTO_COMMIT_INTERVAL_MS");
    let Some(raw) = env_string(&name) else {
        return Ok(default);
    };
    let value = raw
        .parse::<u32>()
        .with_context(|| format!("parse {name} as a positive integer"))?;
    if value == 0 {
        return Err(anyhow!("{name} must be greater than zero"));
    }
    Ok(value)
}

#[derive(Clone)]
pub struct KafkaKvEventProducer {
    producer: FutureProducer,
    topic: Arc<String>,
}

impl KafkaKvEventProducer {
    pub fn new(config: KafkaKvEventStreamConfig) -> Result<Self> {
        let producer = config
            .base_client_config()
            .create()
            .context("create Kafka KV event producer")?;
        Ok(Self {
            producer,
            topic: Arc::new(config.topic),
        })
    }

    pub async fn send(&self, record: &KvEventStreamRecord) -> Result<()> {
        let key = format!("{}:{}", record.worker_url, record.dp_rank);
        let payload = serde_json::to_string(record).context("serialize Kafka KV event record")?;
        self.producer
            .send(
                FutureRecord::to(self.topic.as_str())
                    .key(&key)
                    .payload(&payload),
                Duration::from_secs(5),
            )
            .await
            .map_err(|(err, _)| anyhow!("send Kafka KV event record: {err}"))?;
        Ok(())
    }

    pub fn topic(&self) -> &str {
        self.topic.as_str()
    }
}

#[derive(Clone, Default)]
struct CacheStateKafkaConsumerContext;

impl ClientContext for CacheStateKafkaConsumerContext {}

impl ConsumerContext for CacheStateKafkaConsumerContext {
    fn commit_callback(&self, result: KafkaResult<()>, offsets: &TopicPartitionList) {
        match result {
            Ok(()) => tracing::debug!(
                offsets = ?offsets,
                "periodically committed cache-state Kafka checkpoints"
            ),
            Err(error) => tracing::warn!(
                error = %error,
                offsets = ?offsets,
                "failed to periodically commit cache-state Kafka checkpoints"
            ),
        }
    }
}

type CacheStateStreamConsumer = StreamConsumer<CacheStateKafkaConsumerContext>;

pub struct KafkaKvEventConsumer {
    consumer: CacheStateStreamConsumer,
    topic: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingKafkaRecord {
    pub record: KvEventStreamRecord,
    pub topic: String,
    pub partition: i32,
    pub offset: i64,
}

impl KafkaKvEventConsumer {
    pub fn new(config: KafkaKvEventStreamConfig) -> Result<Self> {
        let group = config
            .consumer_group
            .clone()
            .ok_or_else(|| anyhow!("Kafka consumer group is required"))?;
        let topic = config.topic.clone();
        let client_config = config.consumer_client_config(&group);
        let consumer: CacheStateStreamConsumer = client_config
            .create_with_context(CacheStateKafkaConsumerContext)
            .context("create Kafka KV event consumer")?;
        consumer
            .subscribe(&[&topic])
            .with_context(|| format!("subscribe Kafka topic {topic}"))?;
        Ok(Self { consumer, topic })
    }

    pub async fn recv(&self) -> Result<PendingKafkaRecord> {
        let message = self
            .consumer
            .recv()
            .await
            .context("receive Kafka KV event")?;
        let record = decode_kafka_record(&message)?;
        Ok(PendingKafkaRecord {
            record,
            topic: message.topic().to_string(),
            partition: message.partition(),
            offset: message.offset(),
        })
    }

    pub fn checkpoint(&self, pending: &PendingKafkaRecord) -> Result<()> {
        let next_offset = next_kafka_offset(pending.offset)?;
        let mut offsets = TopicPartitionList::new();
        offsets
            .add_partition_offset(
                &pending.topic,
                pending.partition,
                Offset::Offset(next_offset),
            )
            .context("build Kafka checkpoint offset")?;
        self.consumer
            .store_offsets(&offsets)
            .context("store applied Kafka KV event checkpoint")
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }
}

fn next_kafka_offset(offset: i64) -> Result<i64> {
    offset
        .checked_add(1)
        .ok_or_else(|| anyhow!("Kafka offset overflow at {offset}"))
}

pub fn apply_then_checkpoint<T>(
    pending: &PendingKafkaRecord,
    apply: impl FnOnce(&KvEventStreamRecord) -> Result<T>,
    checkpoint: impl FnOnce(&PendingKafkaRecord) -> Result<()>,
) -> Result<T> {
    let result = apply(&pending.record).context("apply Kafka KV event record")?;
    checkpoint(pending).context("checkpoint Kafka KV event after apply")?;
    Ok(result)
}

pub async fn apply_then_checkpoint_blocking<T, Apply, Checkpoint>(
    pending: Arc<PendingKafkaRecord>,
    apply: Apply,
    checkpoint: Checkpoint,
) -> Result<T>
where
    T: Send + 'static,
    Apply: FnOnce(&KvEventStreamRecord) -> Result<T> + Send + 'static,
    Checkpoint: FnOnce(&PendingKafkaRecord) -> Result<()> + Send + 'static,
{
    tokio::task::spawn_blocking(move || apply_then_checkpoint(pending.as_ref(), apply, checkpoint))
        .await
        .context("join blocking Kafka KV event apply-and-checkpoint task")?
}

impl LocalKvEventStream {
    pub fn new(config: LocalKvEventStreamConfig) -> Self {
        Self {
            config: Arc::new(config),
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn append(&self, record: &KvEventStreamRecord) -> Result<AppendStats> {
        let _guard = self.lock.lock();
        ensure_parent_dir(&self.config.path)?;
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.config.path)
                .with_context(|| format!("open {}", self.config.path.display()))?;
            serde_json::to_writer(&mut file, record).context("serialize KV event stream record")?;
            file.write_all(b"\n")
                .context("write stream record newline")?;
            file.flush().context("flush stream record")?;
        }
        self.compact_locked()
    }

    pub fn read_all(&self) -> Result<Vec<KvEventStreamRecord>> {
        let _guard = self.lock.lock();
        read_records(&self.config.path)
    }

    fn compact_locked(&self) -> Result<AppendStats> {
        let mut records = read_records(&self.config.path)?;
        if let Some(retention_secs) = self.config.retention_secs {
            let min_observed_at_ms = now_ms().saturating_sub(retention_secs.saturating_mul(1000));
            records.retain(|r| r.observed_at_ms >= min_observed_at_ms);
        }
        let mut encoded = encode_records(&records)?;
        if let Some(max_bytes) = self.config.max_bytes {
            while !records.is_empty() && encoded.len() as u64 > max_bytes {
                records.remove(0);
                encoded = encode_records(&records)?;
            }
        }
        write_records_atomically(&self.config.path, &encoded)?;
        Ok(AppendStats {
            records_after_compaction: records.len(),
            bytes_after_compaction: encoded.len() as u64,
        })
    }
}

fn read_records(path: &Path) -> Result<Vec<KvEventStreamRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut records = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("read {} line {}", path.display(), idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str(&line)
                .with_context(|| format!("decode {} line {}", path.display(), idx + 1))?,
        );
    }
    Ok(records)
}

fn encode_records(records: &[KvEventStreamRecord]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for record in records {
        serde_json::to_writer(&mut out, record).context("serialize compacted stream record")?;
        out.push(b'\n');
    }
    Ok(out)
}

fn write_records_atomically(path: &Path, encoded: &[u8]) -> Result<()> {
    ensure_parent_dir(path)?;
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, encoded).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "rename compacted stream {} to {}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    Ok(())
}

fn decode_kafka_record(message: &BorrowedMessage<'_>) -> Result<KvEventStreamRecord> {
    let payload = message
        .payload()
        .ok_or_else(|| anyhow!("Kafka KV event message has empty payload"))?;
    serde_json::from_slice(payload).context("decode Kafka KV event record")
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.trim().is_empty())
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() >= 2 {
            out.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() == 3 {
            out.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn append_and_replay_records() {
        let dir = tempfile::tempdir().unwrap();
        let stream = LocalKvEventStream::new(LocalKvEventStreamConfig {
            path: dir.path().join("kv-events.jsonl"),
            retention_secs: None,
            max_bytes: None,
        });
        let record = KvEventStreamRecord::from_payload(
            "m".into(),
            "http://w0:30000".into(),
            1,
            7,
            b"payload",
        );
        let stats = stream.append(&record).unwrap();
        assert_eq!(stats.records_after_compaction, 1);
        assert_eq!(stream.read_all().unwrap(), vec![record]);
    }

    #[test]
    fn apply_failure_never_invokes_checkpoint() {
        let pending = PendingKafkaRecord {
            record: KvEventStreamRecord::from_payload(
                "m".into(),
                "http://worker".into(),
                0,
                1,
                b"payload",
            ),
            topic: "events".into(),
            partition: 2,
            offset: 41,
        };
        let checkpoints = AtomicUsize::new(0);
        let result: Result<()> = apply_then_checkpoint(
            &pending,
            |_| Err(anyhow!("apply failed")),
            |_| {
                checkpoints.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(checkpoints.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn successful_apply_checkpoints_once_after_apply_and_offsets_advance_by_one() {
        let pending = PendingKafkaRecord {
            record: KvEventStreamRecord::from_payload(
                "m".into(),
                "http://worker".into(),
                0,
                1,
                b"payload",
            ),
            topic: "events".into(),
            partition: 2,
            offset: 41,
        };
        let applied = AtomicBool::new(false);
        let checkpoints = AtomicUsize::new(0);
        let result = apply_then_checkpoint(
            &pending,
            |_| {
                applied.store(true, Ordering::Release);
                Ok(7usize)
            },
            |_| {
                assert!(applied.load(Ordering::Acquire));
                checkpoints.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result, 7);
        assert_eq!(checkpoints.load(Ordering::Relaxed), 1);
        assert_eq!(next_kafka_offset(pending.offset).unwrap(), 42);
        assert!(next_kafka_offset(i64::MAX).is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_apply_and_checkpoint_does_not_starve_single_thread_runtime() {
        let pending = PendingKafkaRecord {
            record: KvEventStreamRecord::from_payload(
                "m".into(),
                "http://worker".into(),
                0,
                1,
                b"payload",
            ),
            topic: "events".into(),
            partition: 2,
            offset: 41,
        };
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let checkpoints = Arc::new(AtomicUsize::new(0));

        let watchdog_started = Arc::clone(&started);
        let watchdog_release = Arc::clone(&release);
        let watchdog = std::thread::spawn(move || {
            while !watchdog_started.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            for _ in 0..100 {
                if watchdog_release.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            watchdog_release.store(true, Ordering::Release);
        });

        let apply_started = Arc::clone(&started);
        let apply_release = Arc::clone(&release);
        let checkpoint_count = Arc::clone(&checkpoints);
        let operation = tokio::spawn(apply_then_checkpoint_blocking(
            Arc::new(pending),
            move |_| {
                apply_started.store(true, Ordering::Release);
                while !apply_release.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Ok(7usize)
            },
            move |_| {
                checkpoint_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        ));

        let wait_started = std::time::Instant::now();
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
        assert!(
            wait_started.elapsed() < Duration::from_millis(500),
            "blocking apply ran on and starved the current-thread runtime"
        );
        release.store(true, Ordering::Release);

        assert_eq!(operation.await.unwrap().unwrap(), 7);
        assert_eq!(checkpoints.load(Ordering::Relaxed), 1);
        watchdog.join().unwrap();
    }

    #[test]
    fn consumer_config_batches_manually_stored_offsets() {
        let config = KafkaKvEventStreamConfig {
            bootstrap_servers: "localhost:9092".into(),
            topic: "events".into(),
            username: None,
            password: None,
            client_id: None,
            consumer_group: Some("cache-state".into()),
            auto_offset_reset: "earliest".into(),
            auto_commit_interval_ms: 1_234,
        };
        let client_config = config.consumer_client_config("cache-state");
        assert_eq!(client_config.get("enable.auto.commit"), Some("true"));
        assert_eq!(client_config.get("enable.auto.offset.store"), Some("false"));
        assert_eq!(client_config.get("auto.commit.interval.ms"), Some("1234"));
    }

    #[test]
    fn retention_drops_old_records() {
        let dir = tempfile::tempdir().unwrap();
        let stream = LocalKvEventStream::new(LocalKvEventStreamConfig {
            path: dir.path().join("kv-events.jsonl"),
            retention_secs: Some(1),
            max_bytes: None,
        });
        let mut old = KvEventStreamRecord::from_payload("m".into(), "w".into(), 0, 1, b"old");
        old.observed_at_ms = now_ms().saturating_sub(5_000);
        let new = KvEventStreamRecord::from_payload("m".into(), "w".into(), 0, 2, b"new");
        stream.append(&old).unwrap();
        stream.append(&new).unwrap();
        assert_eq!(stream.read_all().unwrap(), vec![new]);
    }

    #[test]
    fn max_bytes_drops_oldest_records() {
        let dir = tempfile::tempdir().unwrap();
        let stream = LocalKvEventStream::new(LocalKvEventStreamConfig {
            path: dir.path().join("kv-events.jsonl"),
            retention_secs: None,
            max_bytes: Some(500),
        });
        let first = KvEventStreamRecord::from_payload("m".into(), "w".into(), 0, 1, b"first");
        let second = KvEventStreamRecord::from_payload("m".into(), "w".into(), 0, 2, b"second");
        stream.append(&first).unwrap();
        stream.append(&second).unwrap();
        let replayed = stream.read_all().unwrap();
        assert!(!replayed.is_empty());
        assert_eq!(replayed.last(), Some(&second));
        assert!(
            fs::metadata(dir.path().join("kv-events.jsonl"))
                .unwrap()
                .len()
                <= 500
        );
    }
}
