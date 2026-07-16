"""Unit tests for /v1/loads load snapshot response behavior."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec.msgpack

from sglang.srt.entrypoints.v1_loads import get_loads
from sglang.srt.managers.load_snapshot import (
    HEADER_STRUCT,
    MAGIC,
    SLOT_LEN_STRUCT,
    SLOT_SIZE,
    VERSION,
    LoadSnapshot,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    slot_offset,
)
from sglang.srt.managers.io_struct import (
    PrefillWorkMetrics,
    PrefillWorkOverflowMetrics,
    PrefillWorkRequestMetrics,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


class _FakeTokenizerManager(TokenizerControlMixin):
    def __init__(self, reader, dp_size: int):
        self.load_snapshot_reader = reader
        self.server_args = SimpleNamespace(
            dp_size=dp_size,
            enable_dp_attention=False,
            nnodes=1,
        )

    def auto_create_handle_loop(self):
        pass


class _FakeHttpTokenizerManager:
    metrics_collector = None

    def __init__(self, loads):
        self.loads = loads

    async def get_loads(self, include=None, dp_rank=None):
        results = []
        for load in self.loads:
            if dp_rank is not None and load.dp_rank != dp_rank:
                continue
            results.append(load)
        return results


class _FakeLegacyLoadTokenizerManager:
    def __init__(self):
        self.server_args = SimpleNamespace(disaggregation_mode="prefill")
        self.requested_include = None

    async def get_loads(self, include=None):
        self.requested_include = include
        return [
            SimpleNamespace(
                dp_rank=0,
                num_running_reqs=2,
                num_waiting_reqs=3,
                num_waiting_uncached_tokens=400,
                num_total_tokens=1000,
                num_used_tokens=200,
                has_prefill_queue=1,
                prefill_queue_detail_complete=True,
                prefill_queue_chunked_remaining_uncached_tokens=64,
                prefill_queue_work_bucket_bounds=(256, 1024),
                prefill_queue_priority_scheduling_enabled=True,
                prefill_queue_schedule_low_priority_values_first=False,
                prefill_queue_priority_values=(0,),
                prefill_queue_priority_total_uncached_tokens=(400,),
                prefill_queue_priority_ahead_uncached_tokens=((100, 400),),
            )
        ]


class TestLoadsResponse(CustomTestCase):
    def test_legacy_get_load_requests_and_exports_prefill_queue(self):
        from sglang.srt.entrypoints import http_server

        manager = _FakeLegacyLoadTokenizerManager()
        with patch.object(
            http_server,
            "_global_state",
            SimpleNamespace(tokenizer_manager=manager),
        ):
            response = asyncio.run(http_server.get_load())

        self.assertEqual(
            manager.requested_include, ["core", "prefill_queue", "prefill_work"]
        )
        self.assertEqual(response[0]["num_reqs"], 5)
        self.assertEqual(response[0]["num_running_reqs"], 2)
        self.assertEqual(response[0]["num_waiting_uncached_tokens"], 400)
        self.assertEqual(response[0]["load_role"], "prefill")
        self.assertEqual(
            response[0]["prefill_queue"]["priority_ahead_uncached_tokens"],
            ((100, 400),),
        )

    def test_response_omits_server_side_aggregate_and_redundant_fields(self):
        manager = _FakeHttpTokenizerManager(
            [
                LoadSnapshot(
                    dp_rank=0,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_total_tokens=256,
                )
            ]
        )

        response = asyncio.run(get_loads(tokenizer_manager=manager))

        self.assertNotIn("dp_rank_count", response)
        self.assertNotIn("aggregate", response)
        self.assertEqual(len(response["loads"]), 1)
        self.assertNotIn("num_total_reqs", response["loads"][0])
        self.assertEqual(response["loads"][0]["num_running_reqs"], 3)
        self.assertEqual(response["loads"][0]["num_waiting_reqs"], 2)


class TestGetLoads(CustomTestCase):
    def test_prefill_work_snapshot_round_trips_and_is_optional(self):
        work = PrefillWorkMetrics(
            schema_version=1,
            snapshot_id=42,
            generated_at_ms=1234,
            worker_boot_id="boot-a",
            priority_scheduling_enabled=False,
            schedule_low_priority_values_first=False,
            detail_complete=False,
            truncated=True,
            waiting_prefill=(
                PrefillWorkRequestMetrics(
                    request_id="req-1",
                    priority=100,
                    total_uncached_tokens=4096,
                ),
            ),
            running_prefill=(),
            overflow_summary=(
                PrefillWorkOverflowMetrics(
                    priority=0,
                    length_bucket=8192,
                    request_count=2,
                    total_uncached_tokens=10000,
                ),
            ),
        )
        snapshot = LoadSnapshot(dp_rank=0, prefill_work=work)
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            writer.write(snapshot)
            decoded = reader.read(0)
            self.assertIsNotNone(decoded)
            value = decoded.to_dict({"core", "prefill_work"})
            self.assertEqual(value["prefill_work"]["snapshot_id"], 42)
            self.assertEqual(
                value["prefill_work"]["waiting_prefill"][0]["request_id"],
                "req-1",
            )
            self.assertEqual(
                value["prefill_work"]["overflow_summary"][0]["request_count"], 2
            )
            self.assertNotIn("prefill_work", LoadSnapshot(dp_rank=0).to_dict({"core"}))
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_load_snapshot_wire_format_is_msgpack_slots(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=2, dp_rank=1)
        try:
            writer.write(
                LoadSnapshot(
                    dp_rank=1,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    token_usage=0.25,
                )
            )

            with open(path, "rb") as f:
                data = f.read()

            self.assertEqual(len(data), HEADER_STRUCT.size + 2 * SLOT_SIZE)
            magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(data, 0)
            self.assertEqual(magic, MAGIC)
            self.assertEqual(version, VERSION)
            self.assertEqual(dp_size, 2)
            self.assertEqual(slot_size, SLOT_SIZE)

            offset = slot_offset(1, slot_size)
            (payload_len,) = SLOT_LEN_STRUCT.unpack_from(data, offset)
            payload_start = offset + SLOT_LEN_STRUCT.size
            payload = data[payload_start : payload_start + payload_len]
            decoded = msgspec.msgpack.decode(payload)

            self.assertEqual(decoded["dp_rank"], 1)
            self.assertEqual(decoded["num_running_reqs"], 3)
            self.assertEqual(decoded["num_waiting_reqs"], 2)
            self.assertEqual(decoded["token_usage"], 0.25)
        finally:
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_reads_snapshot_and_filters_sections(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            initial_load = reader.read(0)
            self.assertIsNotNone(initial_load)
            self.assertEqual(initial_load.num_total_tokens, 0)

            writer.write(
                LoadSnapshot(
                    dp_rank=0,
                    timestamp=1.25,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_used_tokens=128,
                    num_total_tokens=256,
                    max_total_num_tokens=4096,
                    token_usage=0.125,
                    gen_throughput=99.5,
                    cache_hit_rate=0.75,
                    utilization=0.5,
                    max_running_requests=128,
                    has_disaggregation=1,
                    disagg_mode=2,
                    decode_transfer_queue_reqs=4,
                    has_queues=1,
                    queue_waiting=2,
                    queue_grammar=1,
                    queue_paused=0,
                    queue_retracted=3,
                    has_prefill_queue=1,
                    prefill_queue_detail_complete=True,
                    prefill_queue_chunked_remaining_uncached_tokens=64,
                    prefill_queue_work_bucket_bounds=(256, 1024),
                    prefill_queue_priority_scheduling_enabled=True,
                    prefill_queue_schedule_low_priority_values_first=False,
                    prefill_queue_priority_values=(0,),
                    prefill_queue_priority_total_uncached_tokens=(500,),
                    prefill_queue_priority_ahead_uncached_tokens=((100, 500),),
                )
            )

            manager = _FakeTokenizerManager(reader, dp_size=1)
            loads = asyncio.run(manager.get_loads(include=["core"], dp_rank=0))

            self.assertEqual(len(loads), 1)
            self.assertEqual(loads[0].num_total_tokens, 256)

            d = loads[0].to_dict({"core"})
            self.assertNotIn("disaggregation", d)
            self.assertNotIn("queues", d)

            loads_all = asyncio.run(manager.get_loads(include=["all"], dp_rank=0))
            d_all = loads_all[0].to_dict()
            self.assertIn("disaggregation", d_all)
            self.assertIn("queues", d_all)
            self.assertEqual(
                d_all["prefill_queue"]["chunked_remaining_uncached_tokens"], 64
            )
            self.assertEqual(
                d_all["prefill_queue"]["priority_ahead_uncached_tokens"],
                ((100, 500),),
            )
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
