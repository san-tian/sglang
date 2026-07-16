import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import PREFILL_QUEUE_PRIORITY_GROUP_LIMIT
from sglang.srt.managers.scheduler_components.load_inquirer import (
    PREFILL_WORK_DETAIL_LIMIT,
    SchedulerLoadInquirer,
    _build_prefill_work_metrics,
    build_prefill_queue_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(
    *,
    tokens,
    matched=0,
    priority=0,
    entry_time=100.0,
    prefix_len=0,
    rid="req",
    extend_end=None,
):
    return SimpleNamespace(
        rid=rid,
        seqlen=tokens,
        num_matched_prefix_tokens=matched,
        priority=priority,
        prefix_indices=range(prefix_len),
        extend_range=(
            SimpleNamespace(end=extend_end) if extend_end is not None else None
        ),
        time_stats=SimpleNamespace(wait_queue_entry_time=entry_time),
    )


class TestPrefillQueueMetrics(CustomTestCase):
    def test_builds_bounded_waiting_and_running_prefill_work(self):
        running = _req(
            rid="running",
            tokens=1000,
            matched=100,
            prefix_len=300,
            extend_end=500,
        )
        metrics = _build_prefill_work_metrics(
            [[_req(rid="waiting", tokens=500, matched=100, priority=10)]],
            [running],
            running,
            priority_scheduling_enabled=True,
            schedule_low_priority_values_first=False,
        )

        self.assertTrue(metrics.detail_complete)
        self.assertFalse(metrics.truncated)
        self.assertEqual(metrics.waiting_prefill[0].request_id, "waiting")
        self.assertEqual(metrics.waiting_prefill[0].total_uncached_tokens, 400)
        self.assertEqual(metrics.running_prefill[0].processed_uncached_tokens, 200)
        self.assertEqual(metrics.running_prefill[0].current_chunk_end_tokens, 400)

    def test_prefill_work_overflow_is_aggregated(self):
        waiting = [
            _req(rid=f"req-{index}", tokens=1000, priority=0)
            for index in range(PREFILL_WORK_DETAIL_LIMIT + 2)
        ]
        metrics = _build_prefill_work_metrics(
            [waiting],
            [],
            None,
            priority_scheduling_enabled=True,
            schedule_low_priority_values_first=False,
        )

        self.assertFalse(metrics.detail_complete)
        self.assertTrue(metrics.truncated)
        self.assertEqual(len(metrics.waiting_prefill), PREFILL_WORK_DETAIL_LIMIT)
        self.assertEqual(metrics.overflow_summary[0].request_count, 2)
        self.assertEqual(metrics.overflow_summary[0].total_uncached_tokens, 2000)

    def test_chunked_request_is_included_when_outside_running_batch(self):
        chunked = _req(
            rid="chunked",
            tokens=1000,
            matched=100,
            prefix_len=300,
            extend_end=500,
        )
        metrics = _build_prefill_work_metrics(
            [],
            [],
            chunked,
            priority_scheduling_enabled=False,
            schedule_low_priority_values_first=False,
        )

        self.assertEqual(len(metrics.running_prefill), 1)
        self.assertEqual(metrics.running_prefill[0].request_id, "chunked")
        self.assertEqual(metrics.running_prefill[0].processed_uncached_tokens, 200)
        self.assertEqual(metrics.running_prefill[0].current_chunk_end_tokens, 400)

    def test_native_prefill_total_includes_bootstrap_waiting_and_chunked_work(self):
        inquirer = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            get_waiting_queue=lambda: [_req(tokens=100, matched=25)],
            get_disagg_prefill_bootstrap_queue=lambda: SimpleNamespace(
                queue=[_req(tokens=200, matched=50)]
            ),
            get_chunked_req=lambda: _req(tokens=300, prefix_len=100),
        )

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer), 425
        )

    def test_builds_candidate_work_ahead_and_chunk_remainder(self):
        metrics = build_prefill_queue_metrics(
            [
                _req(tokens=100, priority=0),
                _req(tokens=1000, priority=0),
                _req(tokens=500, priority=0, entry_time=60.0),
                _req(tokens=300, matched=100, priority=10),
            ],
            _req(tokens=200, prefix_len=50),
            now=100.0,
            aging_rate=0.0,
            max_wait_seconds=30.0,
            priority_scheduling_enabled=True,
            schedule_low_priority_values_first=False,
        )

        self.assertTrue(metrics.detail_complete)
        self.assertEqual(metrics.chunked_remaining_uncached_tokens, 150)
        self.assertEqual(metrics.priority_values, (0, 10))
        self.assertEqual(metrics.priority_total_uncached_tokens, (1600, 200))
        self.assertEqual(metrics.priority_ahead_uncached_tokens[0][0], 600)
        self.assertEqual(metrics.priority_ahead_uncached_tokens[0][1], 1600)
        self.assertEqual(metrics.priority_ahead_uncached_tokens[1][0], 200)

    def test_excess_priority_cardinality_marks_detail_incomplete(self):
        waiting = [
            _req(tokens=1, priority=priority)
            for priority in range(PREFILL_QUEUE_PRIORITY_GROUP_LIMIT + 1)
        ]

        metrics = build_prefill_queue_metrics(
            waiting,
            None,
            now=100.0,
            aging_rate=0.0,
            max_wait_seconds=30.0,
            priority_scheduling_enabled=True,
            schedule_low_priority_values_first=False,
        )

        self.assertFalse(metrics.detail_complete)
        self.assertEqual(metrics.priority_values, ())
        self.assertEqual(metrics.priority_total_uncached_tokens, ())
        self.assertEqual(metrics.priority_ahead_uncached_tokens, ())


if __name__ == "__main__":
    unittest.main()
