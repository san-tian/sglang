import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    SchedulePolicy,
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


class TestRetiredPrefillLengthAwareSchedulePolicy(CustomTestCase):
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

    def test_legacy_policy_maps_to_fcfs(self):
        self.assertEqual(self.policy().policy, CacheAgnosticPolicy.FCFS)
        fcfs = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(fcfs.policy, CacheAgnosticPolicy.FCFS)

    def test_legacy_policy_keeps_fcfs_order(self):
        waiting = [
            _req("oldest-long", 800, entry_time=90.0),
            _req("short-1", 100),
            _req("short-2", 200),
            _req("short-3", 300),
            _req("short-4", 400),
            _req("next-oldest", 700, entry_time=95.0),
            _req("short-5", 50),
        ]

        self.policy().calc_priority(waiting)

        self.assertEqual(
            [req.rid for req in waiting],
            [
                "oldest-long",
                "short-1",
                "short-2",
                "short-3",
                "short-4",
                "next-oldest",
                "short-5",
            ],
        )

    def test_business_priority_preserves_fcfs_within_each_group(self):
        waiting = [
            _req("low-old", 1000, priority=0, entry_time=90.0),
            _req("high-old", 1000, priority=10, entry_time=90.0),
            _req("high-new-short", 1, priority=10, entry_time=100.0),
        ]
        policy = SchedulePolicy(
            policy="prefill-length-aware",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )

        policy.calc_priority(waiting)
        self.assertEqual(
            [req.rid for req in waiting],
            ["high-old", "high-new-short", "low-old"],
        )

        policy.schedule_low_priority_values_first = True
        policy.priority_sign = 1
        policy.calc_priority(waiting)
        self.assertEqual(
            [req.rid for req in waiting],
            ["low-old", "high-old", "high-new-short"],
        )

    def test_calc_priority_refreshes_live_cache_match_before_sorting(self):
        self.tree_cache.supports_fast_match_prefix.return_value = True
        waiting = [
            _req("oldest", 10, entry_time=90.0),
            _req("short", 100),
            _req("cached-long", 1000),
        ]

        def refresh_match(_tree_cache, req, **_kwargs):
            req.num_matched_prefix_tokens = 950 if req.rid == "cached-long" else 0

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
            self.policy().calc_priority(waiting)

        self.assertEqual(match_prefix.call_count, 3)
        self.assertEqual(
            [req.rid for req in waiting], ["oldest", "short", "cached-long"]
        )

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
