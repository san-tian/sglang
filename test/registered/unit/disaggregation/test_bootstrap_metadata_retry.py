"""Unit tests for disaggregation bootstrap metadata send retry."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
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


class _TestKVReceiver(CommonKVReceiver):
    def poll(self):
        raise NotImplementedError


class _ReregisteringTestKVReceiver(_TestKVReceiver):
    def _should_reregister_kv_args_on_cache_hit(self):
        return True


def _make_receiver(receiver_cls=_TestKVReceiver):
    receiver = receiver_cls.__new__(receiver_cls)
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

    def test_retry_refreshes_bootstrap_info_before_metadata_retry(self):
        receiver = _make_receiver()
        old_info = _bootstrap_info(38931)
        receiver.bootstrap_infos = [old_info]
        receiver.kv_mgr.connection_pool["10.60.0.8:8998_0_0_0"] = [old_info]
        refreshed_info = _bootstrap_info(30100)
        events = []

        def send(info, _frames):
            events.append(("send", info))
            if info is old_info:
                raise zmq.Again()

        def refresh_hook(info):
            events.append(("register", info))
            return True

        with (
            patch.object(
                receiver,
                "_send_multipart_to_bootstrap",
                side_effect=send,
            ),
            patch.object(
                receiver,
                "_get_bootstrap_info_from_server",
                return_value=refreshed_info,
            ),
            patch.object(
                receiver,
                "_on_bootstrap_info_refreshed",
                side_effect=refresh_hook,
            ) as mock_refresh_hook,
        ):
            ok = receiver._send_request_multipart_to_bootstrap(old_info, [b"frame"])

        self.assertTrue(ok)
        mock_refresh_hook.assert_called_once_with(refreshed_info)
        self.assertEqual(
            events,
            [
                ("send", old_info),
                ("register", refreshed_info),
                ("send", refreshed_info),
            ],
        )

    def test_mori_refresh_hook_registers_peer_without_nested_refresh(self):
        try:
            from sglang.srt.disaggregation.mori.conn import MoriKVReceiver
        except ImportError as error:
            self.skipTest(f"Mori runtime is unavailable: {error}")

        class _Packed:
            def __init__(self, value):
                self.value = value

            def pack(self):
                return self.value

        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        refreshed_info = _bootstrap_info(30100)
        receiver.bootstrap_infos = [refreshed_info]
        receiver.kv_mgr = SimpleNamespace(
            engine_desc=_Packed(b"engine"),
            kv_mem_descs=[],
            aux_mem_descs=[],
            state_mem_descs=[],
            local_ip="10.60.0.37",
            rank_port=39001,
            attn_tp_size=4,
            kv_args=SimpleNamespace(
                gpu_id=0,
                engine_rank=2,
                kv_item_lens=[128],
                state_item_lens=[],
                state_dim_per_tensor=[],
            ),
        )

        with patch.object(
            receiver,
            "_register_kv_args_to_bootstrap_info",
            return_value=True,
        ) as mock_register:
            ok = receiver._on_bootstrap_info_refreshed(refreshed_info)

        self.assertTrue(ok)
        self.assertIs(mock_register.call_args.args[0], refreshed_info)
        self.assertFalse(
            mock_register.call_args.kwargs["retry_with_fresh_bootstrap_info"]
        )

    def test_mori_metadata_does_not_duplicate_peer_registration(self):
        try:
            from sglang.srt.disaggregation.mori.conn import MoriKVReceiver
        except ImportError as error:
            self.skipTest(f"Mori runtime is unavailable: {error}")

        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        receiver.bootstrap_infos = [_bootstrap_info(30100)]
        receiver.bootstrap_room = 123
        receiver.required_dst_info_num = 1
        receiver.init_time = None
        receiver.kv_mgr = SimpleNamespace(
            local_ip="10.60.0.37",
            rank_port=39001,
            engine_desc=SimpleNamespace(key="decode-engine"),
        )

        with (
            patch.object(receiver, "_register_kv_args") as mock_register,
            patch.object(
                receiver,
                "_send_request_multipart_to_bootstrap",
                return_value=True,
            ) as mock_send,
        ):
            receiver.send_metadata(np.asarray([1, 2], dtype=np.int32))

        mock_register.assert_not_called()
        mock_send.assert_called_once()

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
        receiver = _make_receiver(_ReregisteringTestKVReceiver)
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

    def test_cached_bootstrap_infos_do_not_register_by_default(self):
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

        mock_register.assert_not_called()
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

    def test_get_bootstrap_info_stops_after_three_attempts(self):
        receiver = _make_receiver()

        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                side_effect=ConnectionRefusedError("refused"),
            ) as mock_get,
            patch("sglang.srt.disaggregation.common.conn.time.sleep") as mock_sleep,
        ):
            info = receiver._get_bootstrap_info_from_server(0, 0, 0, 0)

        self.assertIsNone(info)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
