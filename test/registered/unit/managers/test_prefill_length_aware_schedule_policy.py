import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    SchedulePolicy,
    prefill_length_aware_request_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(
    rid: str,
    tokens: int,
    *,
    matched: int = 0,
    priority: int = 0,
    entry_time: float = 100.0,
):
    return SimpleNamespace(
        rid=rid,
        seqlen=tokens,
        num_matched_prefix_tokens=matched,
        priority=priority,
        time_stats=SimpleNamespace(wait_queue_entry_time=entry_time),
    )


class TestPrefillLengthAwareSchedulePolicy(CustomTestCase):
    def setUp(self):
        self.tree_cache = MagicMock()
        self.tree_cache.disable = False
        self.tree_cache.supports_fast_match_prefix.return_value = False

    def policy(self, **kwargs) -> SchedulePolicy:
        return SchedulePolicy(
            policy="prefill-length-aware",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            **kwargs,
        )

    def test_policy_is_opt_in(self):
        self.assertEqual(self.policy().policy, CacheAgnosticPolicy.PREFILL_LENGTH_AWARE)
        fcfs = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(fcfs.policy, CacheAgnosticPolicy.FCFS)

    def test_state_uses_current_cache_match_and_aging(self):
        uncached, effective, overdue = prefill_length_aware_request_state(
            _req("cached", 1000, matched=900, entry_time=95.0),
            now=100.0,
            aging_rate=10.0,
            max_wait_seconds=30.0,
        )

        self.assertEqual(uncached, 100)
        self.assertEqual(effective, 50.0)
        self.assertFalse(overdue)

    @patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=100.0)
    def test_short_first_with_finite_aging(self, _mock_now):
        waiting = [
            _req("long-fresh", 800),
            _req("short", 100),
            _req("long-aged", 1000, entry_time=96.0),
        ]

        self.policy(
            prefill_length_aware_aging_rate=250.0,
            prefill_length_aware_max_wait_seconds=30.0,
        )._sort_by_prefill_length_aware(waiting)

        self.assertEqual(
            [req.rid for req in waiting],
            ["long-aged", "short", "long-fresh"],
        )

    @patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=100.0)
    def test_overdue_requests_use_same_priority_fcfs(self, _mock_now):
        waiting = [
            _req("fresh", 1),
            _req("newer-overdue", 10, entry_time=65.0),
            _req("older-overdue", 1000, entry_time=60.0),
        ]

        self.policy(
            prefill_length_aware_aging_rate=0.0,
            prefill_length_aware_max_wait_seconds=30.0,
        )._sort_by_prefill_length_aware(waiting)

        self.assertEqual(
            [req.rid for req in waiting],
            ["older-overdue", "newer-overdue", "fresh"],
        )

    @patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=100.0)
    def test_business_priority_is_outermost_in_both_directions(self, _mock_now):
        waiting = [_req("low-short", 1), _req("high-long", 1000, priority=10)]
        policy = SchedulePolicy(
            policy="prefill-length-aware",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )

        policy._sort_by_prefill_length_aware(waiting)
        self.assertEqual([req.rid for req in waiting], ["high-long", "low-short"])

        policy.schedule_low_priority_values_first = True
        policy.priority_sign = 1
        policy._sort_by_prefill_length_aware(waiting)
        self.assertEqual([req.rid for req in waiting], ["low-short", "high-long"])

    def test_calc_priority_refreshes_live_cache_match_before_sorting(self):
        self.tree_cache.supports_fast_match_prefix.return_value = True
        waiting = [_req("short", 100), _req("cached-long", 1000)]

        def refresh_match(_tree_cache, req, **_kwargs):
            req.num_matched_prefix_tokens = 950 if req.rid == "cached-long" else 0

        with (
            patch(
                "sglang.srt.managers.schedule_policy.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.managers.schedule_policy.get_global_server_args",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch(
                "sglang.srt.managers.schedule_policy.match_prefix_for_req",
                side_effect=refresh_match,
            ) as match_prefix,
        ):
            self.policy().calc_priority(waiting)

        self.assertEqual(match_prefix.call_count, 2)
        self.assertEqual([req.rid for req in waiting], ["cached-long", "short"])

    def test_fcfs_keeps_order_while_refreshing_load_snapshot_matches(self):
        self.tree_cache.supports_fast_match_prefix.return_value = True
        waiting = [_req("first", 1000), _req("second", 100)]
        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

        def refresh_match(_tree_cache, req, **_kwargs):
            req.num_matched_prefix_tokens = 900 if req.rid == "first" else 0

        with (
            patch(
                "sglang.srt.managers.schedule_policy.get_global_server_args",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch(
                "sglang.srt.managers.schedule_policy.match_prefix_for_req",
                side_effect=refresh_match,
            ) as match_prefix,
        ):
            policy.calc_priority(waiting)

        self.assertEqual(match_prefix.call_count, 2)
        self.assertEqual([req.rid for req in waiting], ["first", "second"])
        self.assertEqual(waiting[0].num_matched_prefix_tokens, 900)


if __name__ == "__main__":
    unittest.main()
