"""Unit tests for disaggregation bootstrap metadata send retry."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import zmq

from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeKVManager:
    def __init__(self):
        self.connection_pool = {}
        self.connection_lock = threading.Lock()
        self.is_mla_backend = False
        self.failures = []
        self.status_updates = []

    def record_failure(self, bootstrap_room, failure_reason):
        self.failures.append((bootstrap_room, failure_reason))

    def update_status(self, bootstrap_room, status):
        self.status_updates.append((bootstrap_room, status))


def _make_receiver():
    receiver = CommonKVReceiver.__new__(CommonKVReceiver)
    receiver.bootstrap_room = 123
    receiver.bootstrap_addr = "10.60.0.8:8998"
    receiver.kv_mgr = _FakeKVManager()
    receiver.conclude_state = None
    return receiver


def _bootstrap_info(rank_port):
    return {
        "rank_ip": "10.60.0.8",
        "rank_port": rank_port,
        "is_dummy": False,
        "_prefill_dp_rank": 0,
        "_prefill_cp_rank": 0,
        "_target_tp_rank": 0,
        "_target_pp_rank": 0,
        "_bootstrap_key": "10.60.0.8:8998_0_0_0",
    }


class TestBootstrapMetadataRetry(CustomTestCase):
    def test_retry_refreshes_only_failed_bootstrap_endpoint(self):
        receiver = _make_receiver()
        old_info = _bootstrap_info(38931)
        receiver.bootstrap_infos = [old_info]
        receiver.kv_mgr.connection_pool["10.60.0.8:8998_0_0_0"] = [old_info]
        refreshed_info = _bootstrap_info(30100)

        with (
            patch.object(
                receiver,
                "_send_multipart_to_bootstrap",
                side_effect=[zmq.Again(), None],
            ) as mock_send,
            patch.object(
                receiver,
                "_get_bootstrap_info_from_server",
                return_value=refreshed_info,
            ) as mock_route,
        ):
            ok = receiver._send_request_multipart_to_bootstrap(old_info, [b"frame"])

        self.assertTrue(ok)
        self.assertEqual(mock_send.call_count, 2)
        self.assertIs(mock_send.call_args_list[0].args[0], old_info)
        self.assertIs(mock_send.call_args_list[1].args[0], refreshed_info)
        mock_route.assert_called_once_with(0, 0, 0, 0)
        self.assertIs(receiver.bootstrap_infos[0], refreshed_info)
        self.assertIs(
            receiver.kv_mgr.connection_pool["10.60.0.8:8998_0_0_0"][0],
            refreshed_info,
        )
        self.assertEqual(receiver.kv_mgr.failures, [])

    def test_retry_failure_records_both_endpoints(self):
        receiver = _make_receiver()
        old_info = _bootstrap_info(38931)
        receiver.bootstrap_infos = [old_info]
        refreshed_info = _bootstrap_info(37251)

        with (
            patch.object(
                receiver,
                "_send_multipart_to_bootstrap",
                side_effect=[zmq.Again(), zmq.Again()],
            ),
            patch.object(
                receiver,
                "_get_bootstrap_info_from_server",
                return_value=refreshed_info,
            ),
        ):
            ok = receiver._send_request_multipart_to_bootstrap(old_info, [b"frame"])

        self.assertFalse(ok)
        self.assertEqual(len(receiver.kv_mgr.failures), 1)
        _, reason = receiver.kv_mgr.failures[0]
        self.assertIn("tcp://10.60.0.8:38931", reason)
        self.assertIn("tcp://10.60.0.8:37251", reason)

    def test_cached_bootstrap_infos_still_registers_kv_args(self):
        receiver = _make_receiver()
        cached_info = _bootstrap_info(30100)
        receiver.kv_mgr.connection_pool["10.60.0.8:8998_0_0_0"] = [cached_info]
        receiver.prefill_dp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver.target_tp_rank = 0
        receiver.target_tp_ranks = [0]
        receiver.target_pp_ranks = [0]

        with patch.object(receiver, "_register_kv_args") as mock_register:
            receiver._setup_bootstrap_infos()

        mock_register.assert_called_once_with()
        self.assertEqual(receiver.bootstrap_infos, [cached_info])

    def test_get_bootstrap_info_retries_transient_route_failure(self):
        receiver = _make_receiver()
        response = SimpleNamespace(status_code=200, json=lambda: {"rank_ip": "x"})

        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                side_effect=[ConnectionRefusedError("refused"), response],
            ) as mock_get,
            patch("sglang.srt.disaggregation.common.conn.time.sleep") as mock_sleep,
        ):
            info = receiver._get_bootstrap_info_from_server(0, 0, 0, 0)

        self.assertEqual(info, {"rank_ip": "x"})
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()

    def test_get_bootstrap_info_retries_transient_http_status(self):
        receiver = _make_receiver()
        busy = SimpleNamespace(status_code=503, text="busy")
        ok = SimpleNamespace(status_code=200, json=lambda: {"rank_ip": "x"})

        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                side_effect=[busy, ok],
            ) as mock_get,
            patch("sglang.srt.disaggregation.common.conn.time.sleep") as mock_sleep,
        ):
            info = receiver._get_bootstrap_info_from_server(0, 0, 0, 0)

        self.assertEqual(info, {"rank_ip": "x"})
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()


if __name__ == "__main__":
    import unittest

    unittest.main()
