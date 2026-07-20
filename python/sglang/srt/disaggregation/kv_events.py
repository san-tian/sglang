"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
KV caching events
"""

import atexit
import enum
import hashlib
import logging
import queue
import struct
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from queue import Queue
from typing import Any, Callable, Optional, Union

import msgspec
import zmq
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def select_kv_publisher_dp_rank(
    attn_dp_size: int, attn_dp_rank: int, dp_rank: Optional[int]
) -> int:
    """Index used to offset this scheduler's KV-event publisher port.

    Each independent KV cache must publish on its own port so a consumer can
    subscribe per replica. There are always ``dp_size`` such publishers; which
    rank distinguishes them depends on the parallelism mode:

    - DP-attention (``attn_dp_size > 1``): each attention-DP rank owns a KV
      cache shard, so distinguish by ``attn_dp_rank``.
    - Pure DP (``attn_dp_size == 1``): every worker has ``attn_dp_rank == 0``,
      so distinguish by ``dp_rank`` (the data-parallel replica index).

    Both span ``0..dp_size-1``, matching the ``dp_size`` advertised in
    ``/server_info`` and the per-rank ports the router subscribes to.
    """
    if attn_dp_size > 1:
        return attn_dp_rank
    return dp_rank or 0


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    attn_dp_rank: Optional[int] = None


class CacheStateReconciliationRecord(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for worker-authoritative cache-state control records."""


class CacheStateDigest(CacheStateReconciliationRecord):
    through_seq: int
    block_count: int
    algorithm: str
    digest: str


class CacheStateSnapshotEntry(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    parent_block_hash: Optional[int]
    block_hash: int
    media: list[str]


class CacheStateSnapshotStart(CacheStateReconciliationRecord):
    snapshot_id: str
    watermark: int
    total_chunks: int
    block_count: int
    algorithm: str
    digest: str


class CacheStateSnapshotChunk(CacheStateReconciliationRecord):
    snapshot_id: str
    chunk_index: int
    entries: list[CacheStateSnapshotEntry]


class CacheStateSnapshotEnd(CacheStateReconciliationRecord):
    snapshot_id: str
    watermark: int
    total_chunks: int
    block_count: int
    algorithm: str
    digest: str


class ReconciliationEventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """Opt-in extension of the legacy three-field ``EventBatch`` wire shape."""

    ts: float
    events: list[Any]
    attn_dp_rank: Optional[int]
    publisher_epoch: str
    reconciliation: Optional[
        Union[
            CacheStateDigest,
            CacheStateSnapshotStart,
            CacheStateSnapshotChunk,
            CacheStateSnapshotEnd,
        ]
    ]


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


class StorageMedium(str, enum.Enum):
    """Storage tier for KV cache events."""

    GPU = "GPU"  # L1: device HBM
    CPU = "CPU_PINNED"  # L2: host pinned memory
    DISK = "DISK"  # L3: SSD / NVMe
    EXTERNAL = "EXTERNAL"  # L4: shared / remote pool (e.g. Mooncake)


class OffloadedState:
    """
    OffloadedState represents the state of a KV cache block offloaded to the hicache.

    - prefill_len (int): The length of the prefill part of the KV cache block.
    - inc_len (int): The length of the incremental part of the KV cache block.
    - last_hash (Optional[str]): The hash of the last token in the KV cache block.
    """

    def __init__(
        self, prefill_len: int, inc_len: int = 0, last_hash: Optional[str] = None
    ):
        self.prefill_len = prefill_len
        self.inc_len = inc_len
        self.last_hash = last_hash


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


class CacheStateTracker:
    """Logical routing state maintained by the publisher thread.

    Entries are keyed by ``(parent_hash, block_hash)`` so identical block
    hashes in different chains remain distinct. Storage media are reference
    counted as a set: removing one tier does not hide an entry that is still
    present in another tier.
    """

    DIGEST_ALGORITHM = "xor-sha256-v1"
    DIGEST_BYTES = 32
    _DEFAULT_MEDIUM = StorageMedium.GPU.value

    def __init__(self) -> None:
        self._media_by_entry: dict[tuple[Optional[int], int], set[str]] = {}
        self._xor_digest = bytearray(self.DIGEST_BYTES)

    @staticmethod
    def canonical_entry_bytes(parent_hash: Optional[int], block_hash: int) -> bytes:
        return struct.pack(
            ">Bqq",
            0 if parent_hash is None else 1,
            0 if parent_hash is None else parent_hash,
            block_hash,
        )

    @classmethod
    def entry_digest(cls, parent_hash: Optional[int], block_hash: int) -> bytes:
        return hashlib.sha256(
            cls.canonical_entry_bytes(parent_hash, block_hash)
        ).digest()

    def _xor_entry(self, parent_hash: Optional[int], block_hash: int) -> None:
        item = self.entry_digest(parent_hash, block_hash)
        for i, value in enumerate(item):
            self._xor_digest[i] ^= value

    @classmethod
    def _medium_key(cls, medium: Optional[str]) -> str:
        if isinstance(medium, StorageMedium):
            return medium.value
        return str(medium) if medium is not None else cls._DEFAULT_MEDIUM

    def apply(self, events: list[Any]) -> None:
        for event in events:
            if isinstance(event, BlockStored):
                medium = self._medium_key(event.medium)
                parent_hash = event.parent_block_hash
                for block_hash in event.block_hashes:
                    key = (parent_hash, block_hash)
                    media = self._media_by_entry.get(key)
                    if media is None:
                        media = set()
                        self._media_by_entry[key] = media
                        self._xor_entry(*key)
                    media.add(medium)
                    parent_hash = block_hash
            elif isinstance(event, BlockRemoved):
                medium = self._medium_key(event.medium)
                hashes = set(event.block_hashes)
                for key in [key for key in self._media_by_entry if key[1] in hashes]:
                    media = self._media_by_entry[key]
                    media.discard(medium)
                    if not media:
                        del self._media_by_entry[key]
                        self._xor_entry(*key)
            elif isinstance(event, AllBlocksCleared):
                self._media_by_entry.clear()
                self._xor_digest = bytearray(self.DIGEST_BYTES)

    @property
    def block_count(self) -> int:
        return len(self._media_by_entry)

    @property
    def digest_hex(self) -> str:
        return self._xor_digest.hex()

    def snapshot_entries(self) -> list[CacheStateSnapshotEntry]:
        keys = sorted(
            self._media_by_entry,
            key=lambda item: (
                0 if item[0] is None else 1,
                0 if item[0] is None else item[0],
                item[1],
            ),
        )
        return [
            CacheStateSnapshotEntry(
                parent,
                block,
                sorted(self._media_by_entry[(parent, block)]),
            )
            for parent, block in keys
        ]


class EventPublisher(ABC):
    """
    Lightweight publisher for EventBatch batches with
    support for DP attention.

    In DP attention - each rank has its own Scheduler and
    KV cache instance in order to avoid duplicate events
    and ensure proper event attribution. In our implementation

    - Each DP rank has its own EventPublisher
    - Publishers annotate events with the dp rank
    - This allows consumers to distinguish events from different DP ranks
    """

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return

    def shutdown(self) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

    Spawns a separate thread to handle publishing from a queue.

    Parameters
    ----------
    endpoint:
        PUB address. Use ``tcp://*:5557`` to bind or ``tcp://host:5557`` to
        connect.
    replay_endpoint:
        Optional ROUTER address for replay requests. When given, subscribers can
        request missed batches by sending the starting sequence number as an
        8-byte big-endian integer.
    buffer_steps:
        Number of past batches to keep for replay.
    hwm:
        ZeroMQ high-water-mark for PUB socket.
    max_queue_size:
        Maximum number of events to buffer in memory.
    topic:
        Topic to publish events to.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        attn_dp_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
        reconciliation_enabled: bool = False,
        reconciliation_digest_interval_s: float = 30.0,
        reconciliation_snapshot_interval_s: float = 600.0,
        reconciliation_snapshot_chunk_bytes: int = 256 * 1024,
        reconciliation_max_snapshot_entries: int = 2_000_000,
    ) -> None:
        if reconciliation_enabled and reconciliation_digest_interval_s <= 0:
            raise ValueError("reconciliation_digest_interval_s must be greater than 0")
        if reconciliation_snapshot_interval_s < 0:
            raise ValueError(
                "reconciliation_snapshot_interval_s must be greater than or equal to 0"
            )
        if reconciliation_snapshot_chunk_bytes < 512:
            raise ValueError("reconciliation_snapshot_chunk_bytes must be at least 512")
        if reconciliation_max_snapshot_entries <= 0:
            raise ValueError(
                "reconciliation_max_snapshot_entries must be greater than 0"
            )

        # Storage
        self._event_queue = Queue[Optional[EventBatch]](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._dp_rank = attn_dp_rank
        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        self._hwm = hwm
        self._socket_setup()

        # Payload
        self._pack = msgspec.msgpack.Encoder()
        self._next_seq = 0
        self._topic_bytes = topic.encode("utf-8")
        self._reconciliation_enabled = reconciliation_enabled
        self._publisher_epoch = uuid.uuid4().hex if reconciliation_enabled else None
        self._tracker = CacheStateTracker() if reconciliation_enabled else None
        self._digest_interval_s = reconciliation_digest_interval_s
        self._snapshot_interval_s = reconciliation_snapshot_interval_s
        self._snapshot_chunk_bytes = reconciliation_snapshot_chunk_bytes
        self._max_snapshot_entries = reconciliation_max_snapshot_entries
        now = time.monotonic()
        self._last_digest_at = now
        self._last_snapshot_at = now

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="zmq-publisher"
        )
        self._thread.start()

        atexit.register(self.shutdown)

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.attn_dp_rank is None:
            events.attn_dp_rank = self._dp_rank
        self._event_queue.put(events)

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        self._running = False
        self._event_queue.put_nowait(None)

        start = time.time()
        pending_items = True
        while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
            pending_items = not self._event_queue.empty()
            if pending_items:
                time.sleep(0.1)

        if pending_items:
            logger.warning(
                "Warning: Queue still has %s items after %s seconds timeout",
                self._event_queue.qsize(),
                self.SHUTDOWN_TIMEOUT,
            )

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        # Clean up ZMQ resources
        try:
            if self._pub is not None:
                self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            pass  # Do not terminate context; other sockets may use it

    def _socket_setup(self) -> None:
        """Initialize sockets
        https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
        """
        if self._pub is None:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.set_hwm(self._hwm)
            # Heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention.
            # ``0.0.0.0`` is the IPv4 bind-all wildcard alongside ``*``
            # and ``::``; ``/server_info`` advertises it as a wildcard,
            # so the publisher must bind it for the advertised endpoint
            # to actually be listening.
            if (
                "*" in self._endpoint
                or "::" in self._endpoint
                or "0.0.0.0" in self._endpoint
                or self._endpoint.startswith("ipc://")
                or self._endpoint.startswith("inproc://")
            ):
                logger.debug(
                    f"ZmqEventPublisher socket publisher_endpoint bind to {self._endpoint}"
                )
                self._pub.bind(self._endpoint)
            else:
                self._pub.connect(self._endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            logger.debug(
                f"ZmqEventPublisher socket replay_endpoint bind to {self._replay_endpoint}"
            )
            self._replay.bind(self._replay_endpoint)

    def _publisher_thread(self) -> None:
        """Background thread that processes the event queue."""
        assert self._pub is not None  # narrows type for mypy

        while self._running or self._event_queue.qsize() > 0:
            # --- replay (non-critical) ---------------------------------
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.exception("Error in replay: %s", e)

            # --- main queue (critical) ---------------------------------
            event: Optional[EventBatch] = None
            dequeued = False
            try:
                event = self._event_queue.get(timeout=self._queue_poll_timeout())
                dequeued = True
                if event is None:
                    self._event_queue.task_done()
                    break  # Sentinel received, exit thread
            except queue.Empty:
                pass

            if event is not None:
                try:
                    if self._tracker is not None:
                        self._tracker.apply(event.events)
                    self._send_mutation_batch(event)
                except Exception as e:
                    # Publishing failed; back off to avoid a tight error loop.
                    logger.exception("Error in publisher thread: %s", e)
                    time.sleep(0.1)
                finally:
                    if dequeued:
                        self._event_queue.task_done()

            if self._reconciliation_enabled:
                try:
                    self._emit_due_reconciliation()
                except Exception as e:
                    logger.exception(
                        "Error publishing cache-state reconciliation: %s", e
                    )
                    time.sleep(0.1)

    def _queue_poll_timeout(self) -> float:
        if not self._reconciliation_enabled:
            return 0.1
        now = time.monotonic()
        due_in = max(0.0, self._digest_interval_s - (now - self._last_digest_at))
        if self._snapshot_interval_s > 0:
            due_in = min(
                due_in,
                max(0.0, self._snapshot_interval_s - (now - self._last_snapshot_at)),
            )
        return min(0.1, due_in)

    def _allocate_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def _send_payload(self, seq: int, payload: bytes) -> None:
        assert self._pub is not None
        self._pub.send_multipart((self._topic_bytes, seq.to_bytes(8, "big"), payload))
        self._buffer.append((seq, payload))

    def _send_mutation_batch(self, event: EventBatch) -> int:
        seq = self._allocate_seq()
        if self._publisher_epoch is None:
            payload = self._pack.encode(event)
        else:
            payload = self._pack.encode(
                ReconciliationEventBatch(
                    ts=event.ts,
                    events=event.events,
                    attn_dp_rank=event.attn_dp_rank,
                    publisher_epoch=self._publisher_epoch,
                    reconciliation=None,
                )
            )
        self._send_payload(seq, payload)
        return seq

    def _send_control(self, seq: int, control: CacheStateReconciliationRecord) -> None:
        assert self._publisher_epoch is not None
        payload = self._pack.encode(
            ReconciliationEventBatch(
                ts=time.time(),
                events=[],
                attn_dp_rank=self._dp_rank,
                publisher_epoch=self._publisher_epoch,
                reconciliation=control,
            )
        )
        self._send_payload(seq, payload)

    def _emit_due_reconciliation(self) -> None:
        assert self._tracker is not None
        now = time.monotonic()
        if now - self._last_digest_at >= self._digest_interval_s:
            seq = self._allocate_seq()
            self._send_control(
                seq,
                CacheStateDigest(
                    through_seq=seq,
                    block_count=self._tracker.block_count,
                    algorithm=CacheStateTracker.DIGEST_ALGORITHM,
                    digest=self._tracker.digest_hex,
                ),
            )
            self._last_digest_at = now

        if (
            self._snapshot_interval_s > 0
            and now - self._last_snapshot_at >= self._snapshot_interval_s
        ):
            self._emit_snapshot()
            self._last_snapshot_at = now

    def _emit_snapshot(self) -> None:
        assert self._tracker is not None
        entries = self._tracker.snapshot_entries()
        if len(entries) > self._max_snapshot_entries:
            logger.error(
                "Skipping cache-state snapshot with %s entries; configured cap is %s",
                len(entries),
                self._max_snapshot_entries,
            )
            return

        snapshot_id = uuid.uuid4().hex
        chunks = self._chunk_snapshot_entries(snapshot_id, entries)
        watermark = self._next_seq
        common = dict(
            snapshot_id=snapshot_id,
            watermark=watermark,
            total_chunks=len(chunks),
            block_count=self._tracker.block_count,
            algorithm=CacheStateTracker.DIGEST_ALGORITHM,
            digest=self._tracker.digest_hex,
        )
        start_seq = self._allocate_seq()
        assert start_seq == watermark
        self._send_control(start_seq, CacheStateSnapshotStart(**common))
        for chunk_index, chunk in enumerate(chunks):
            self._send_control(
                self._allocate_seq(),
                CacheStateSnapshotChunk(
                    snapshot_id=snapshot_id,
                    chunk_index=chunk_index,
                    entries=chunk,
                ),
            )
        self._send_control(self._allocate_seq(), CacheStateSnapshotEnd(**common))

    def _chunk_snapshot_entries(
        self, snapshot_id: str, entries: list[CacheStateSnapshotEntry]
    ) -> list[list[CacheStateSnapshotEntry]]:
        if not entries:
            return []
        assert self._publisher_epoch is not None
        chunks: list[list[CacheStateSnapshotEntry]] = []
        current: list[CacheStateSnapshotEntry] = []
        for entry in entries:
            candidate = [*current, entry]
            encoded = self._pack.encode(
                ReconciliationEventBatch(
                    ts=0.0,
                    events=[],
                    attn_dp_rank=self._dp_rank,
                    publisher_epoch=self._publisher_epoch,
                    reconciliation=CacheStateSnapshotChunk(
                        snapshot_id=snapshot_id,
                        chunk_index=len(chunks),
                        entries=candidate,
                    ),
                )
            )
            if len(encoded) <= self._snapshot_chunk_bytes:
                current = candidate
                continue
            if not current:
                raise ValueError(
                    "reconciliation_snapshot_chunk_bytes is too small for one entry"
                )
            chunks.append(current)
            current = [entry]
            single = self._pack.encode(
                ReconciliationEventBatch(
                    ts=0.0,
                    events=[],
                    attn_dp_rank=self._dp_rank,
                    publisher_epoch=self._publisher_epoch,
                    reconciliation=CacheStateSnapshotChunk(
                        snapshot_id=snapshot_id,
                        chunk_index=len(chunks),
                        entries=current,
                    ),
                )
            )
            if len(single) > self._snapshot_chunk_bytes:
                raise ValueError(
                    "reconciliation_snapshot_chunk_bytes is too small for one entry"
                )
        if current:
            chunks.append(current)
        return chunks

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        assert self._replay is not None  # narrows type for mypy

        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning("Invalid replay request: %s", frame)
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")

        for seq, buf in self._buffer:
            if seq >= start_seq:
                # [identity, empty_delim, seq_bytes, payload]
                # (identity, empty_delim) are stripped off by the router
                # receiving payload is (seq_bytes, payload)
                self._replay.send_multipart(
                    (client_id, b"", seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        # receiving payload is (-1, b""")
        self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(
        endpoint: Optional[str], data_parallel_rank: int
    ) -> Optional[str]:
        """Helper function to offset the port in an endpoint by
            the data parallel rank.

        Args:
            endpoint: The endpoint string
                (e.g., "tcp://*:5557" or "inproc://cache")
            data_parallel_rank: The data parallel rank to offset by

        Returns:
            The endpoint with the port offset by data_parallel_rank
                or suffix appended
        """
        # Do nothing if input is None or data_parallel_rank is 0
        if not endpoint or data_parallel_rank == 0:
            return endpoint

        if "inproc" in endpoint:
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint:
            if endpoint and ":" in endpoint:
                # Get everything after the last colon (the port)
                last_colon_idx = endpoint.rfind(":")
                base_addr = endpoint[:last_colon_idx]
                base_port = int(endpoint[last_colon_idx + 1 :])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


class KVEventsConfig(BaseModel):
    """Configuration for KV event publishing."""

    publisher: str = "null"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: Optional[str] = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    reconciliation_enabled: bool = False
    """Enable worker-authoritative epoch, digest, and snapshot emission."""

    reconciliation_digest_interval_s: float = 30.0
    """Seconds between authoritative digest records while enabled."""

    reconciliation_snapshot_interval_s: float = 600.0
    """Seconds between full routing-metadata snapshots; zero disables snapshots."""

    reconciliation_snapshot_chunk_bytes: int = 256 * 1024
    """Maximum encoded bytes for each snapshot chunk batch."""

    reconciliation_max_snapshot_entries: int = 2_000_000
    """Maximum logical entries allowed in one generated snapshot."""

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVEventsConfig":
        """Parse the CLI value for the event publisher config."""
        return KVEventsConfig.model_validate_json(cli_value)


class EventPublisherFactory:
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str, ctor: Callable[..., EventPublisher]) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(cls, config: Optional[str], attn_dp_rank: int = 0) -> EventPublisher:
        """Create publisher from a config mapping."""
        if not config:
            return NullEventPublisher()
        config = KVEventsConfig.from_cli(config)
        config_dict = config.model_dump()

        kind = config_dict.pop("publisher", "null")
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(attn_dp_rank=attn_dp_rank, **config_dict)
