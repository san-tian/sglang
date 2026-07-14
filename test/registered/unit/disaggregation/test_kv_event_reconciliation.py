import time
import unittest
from unittest.mock import MagicMock, patch

import msgspec

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    CacheStateDigest,
    CacheStateTracker,
    EventBatch,
    ReconciliationEventBatch,
    StorageMedium,
    ZmqEventPublisher,
)


def _stored(block_hash, parent=None, medium=StorageMedium.GPU):
    return BlockStored(
        block_hashes=[block_hash],
        parent_block_hash=parent,
        token_ids=[],
        block_size=1,
        lora_id=None,
        medium=medium,
    )


class TestCacheStateTracker(unittest.TestCase):
    def test_legacy_batch_wire_shape_is_unchanged(self):
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(EventBatch(ts=1.0, events=[], attn_dp_rank=None))
        )
        self.assertEqual(decoded, [1.0, [], None])

    def test_reconciliation_batch_has_epoch_and_tagged_control(self):
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(
                ReconciliationEventBatch(
                    ts=1.0,
                    events=[],
                    attn_dp_rank=2,
                    publisher_epoch="epoch-1",
                    reconciliation=CacheStateDigest(
                        through_seq=7,
                        block_count=0,
                        algorithm="xor-sha256-v1",
                        digest="00" * 32,
                    ),
                )
            )
        )
        self.assertEqual(decoded[:4], [1.0, [], 2, "epoch-1"])
        self.assertEqual(decoded[4][0], "CacheStateDigest")

    def test_medium_union_and_chain_entries(self):
        tracker = CacheStateTracker()
        tracker.apply(
            [
                _stored(10, medium=StorageMedium.GPU),
                _stored(10, medium=StorageMedium.CPU),
                _stored(20, parent=10),
            ]
        )
        digest_before = tracker.digest_hex
        self.assertEqual(tracker.block_count, 2)

        tracker.apply([BlockRemoved([10], StorageMedium.GPU)])
        self.assertEqual(tracker.block_count, 2)
        self.assertEqual(tracker.digest_hex, digest_before)

        tracker.apply([BlockRemoved([10], StorageMedium.CPU)])
        self.assertEqual(tracker.block_count, 1)
        self.assertNotEqual(tracker.digest_hex, digest_before)
        self.assertEqual(
            [
                (entry.parent_block_hash, entry.block_hash, entry.media)
                for entry in tracker.snapshot_entries()
            ],
            [(10, 20, ["GPU"])],
        )

        tracker.apply([AllBlocksCleared()])
        self.assertEqual(tracker.block_count, 0)
        self.assertEqual(tracker.digest_hex, "00" * 32)

    def test_digest_is_order_independent(self):
        a = CacheStateTracker()
        b = CacheStateTracker()
        a.apply([_stored(-3), _stored(7, parent=-3)])
        b.apply([_stored(7, parent=-3), _stored(-3)])
        self.assertEqual(a.block_count, b.block_count)
        self.assertEqual(a.digest_hex, b.digest_hex)


class TestReconciliationPublisher(unittest.TestCase):
    def _publisher(self, **overrides):
        defaults = dict(
            attn_dp_rank=0,
            endpoint="inproc://kv-reconciliation-test",
            reconciliation_enabled=True,
            reconciliation_digest_interval_s=0.01,
            reconciliation_snapshot_interval_s=0,
            reconciliation_snapshot_chunk_bytes=512,
        )
        defaults.update(overrides)
        with patch.object(ZmqEventPublisher, "_socket_setup"):
            publisher = ZmqEventPublisher.__new__(ZmqEventPublisher)
            # Exercise initialization without racing a real background thread.
            with patch("threading.Thread.start"), patch("atexit.register"):
                ZmqEventPublisher.__init__(publisher, **defaults)
        publisher._pub = MagicMock()
        return publisher

    def tearDown(self):
        # Publishers are manually stopped because their mocked threads never run.
        for publisher in getattr(self, "_publishers", []):
            publisher._running = False

    def _track(self, publisher):
        if not hasattr(self, "_publishers"):
            self._publishers = []
        self._publishers.append(publisher)
        return publisher

    def test_idle_digest_uses_control_sequence_as_watermark(self):
        publisher = self._track(self._publisher())
        publisher._last_digest_at = time.monotonic() - 1
        publisher._emit_due_reconciliation()

        self.assertEqual(len(publisher._buffer), 1)
        seq, payload = publisher._buffer[0]
        decoded = msgspec.msgpack.decode(payload)
        self.assertEqual(seq, 0)
        self.assertEqual(decoded[4][0], "CacheStateDigest")
        self.assertEqual(decoded[4][1], seq)

    def test_snapshot_chunks_are_bounded_and_not_interleaved(self):
        publisher = self._track(self._publisher())
        publisher._tracker.apply([_stored(i) for i in range(1000)])
        publisher._emit_snapshot()

        controls = []
        for seq, payload in publisher._buffer:
            self.assertLessEqual(
                len(payload),
                publisher._snapshot_chunk_bytes,
                f"sequence {seq} exceeded the configured chunk bound",
            )
            controls.append(msgspec.msgpack.decode(payload)[4][0])
        self.assertEqual(controls[0], "CacheStateSnapshotStart")
        self.assertEqual(controls[-1], "CacheStateSnapshotEnd")
        self.assertTrue(all(tag == "CacheStateSnapshotChunk" for tag in controls[1:-1]))
        self.assertGreater(len(controls), 3)

        start = msgspec.msgpack.decode(publisher._buffer[0][1])[4]
        end = msgspec.msgpack.decode(publisher._buffer[-1][1])[4]
        self.assertEqual(start[1:], end[1:])

    def test_disabled_publisher_encodes_legacy_batch(self):
        publisher = self._track(
            self._publisher(
                reconciliation_enabled=False,
                reconciliation_snapshot_chunk_bytes=512,
            )
        )
        publisher._send_mutation_batch(EventBatch(1.0, [], 0))
        decoded = msgspec.msgpack.decode(publisher._buffer[0][1])
        self.assertEqual(decoded, [1.0, [], 0])


if __name__ == "__main__":
    unittest.main()
