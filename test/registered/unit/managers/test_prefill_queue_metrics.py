import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import PREFILL_QUEUE_PRIORITY_GROUP_LIMIT
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
    build_prefill_queue_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(*, tokens, matched=0, priority=0, entry_time=100.0, prefix_len=0):
    return SimpleNamespace(
        seqlen=tokens,
        num_matched_prefix_tokens=matched,
        priority=priority,
        prefix_indices=range(prefix_len),
        time_stats=SimpleNamespace(wait_queue_entry_time=entry_time),
    )


class TestPrefillQueueMetrics(CustomTestCase):
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

    def test_builds_priority_bucket_cumulative_work_and_chunk_remainder(self):
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
