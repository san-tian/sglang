"""Unit tests for the worker IO-logging helper (feat-MAC-9867).

Loads the module directly via importlib to avoid the heavy sglang package __init__
(which pulls torch/orjson). Mirrors the relay (macaron-relay-stack) and sgl-router
designs: env-gated full request/response body logging aligned to the v3 parquet
field set.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "sglang" / "srt" / "utils" / "io_log.py"
)


@pytest.fixture(scope="module")
def iol():
    spec = importlib.util.spec_from_file_location(
        "sglang_io_log_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UPSTREAM_IO_LOG", raising=False)
    monkeypatch.delenv("ENV_MODE", raising=False)


def test_disabled_by_default(iol):
    assert iol.io_log_enabled() is False


def test_enabled_by_upstream_io_log(iol, monkeypatch):
    monkeypatch.setenv("UPSTREAM_IO_LOG", "1")
    assert iol.io_log_enabled() is True
    monkeypatch.setenv("UPSTREAM_IO_LOG", "true")
    assert iol.io_log_enabled() is True


def test_enabled_by_env_mode_alpha(iol, monkeypatch):
    monkeypatch.setenv("ENV_MODE", "alpha")
    assert iol.io_log_enabled() is True
    monkeypatch.setenv("ENV_MODE", "ALPHA")
    assert iol.io_log_enabled() is True


def test_env_mode_prod_disables(iol, monkeypatch):
    monkeypatch.setenv("ENV_MODE", "prod")
    assert iol.io_log_enabled() is False


def test_clip_caps_and_marks_truncation(iol, monkeypatch):
    monkeypatch.setenv("UPSTREAM_IO_LOG_MAX_BODY_BYTES", "8")  # head=4, tail=4
    big = b"HEADmiddleTAIL"  # 14 bytes, exceeds cap
    clipped, truncated = iol._clip(big)
    assert truncated is True
    assert "HEAD" in clipped, f"head retained: {clipped}"
    assert "TAIL" in clipped, f"tail retained: {clipped}"
    assert "truncated 6 bytes" in clipped, f"marker names dropped count: {clipped}"


def test_clip_preserves_small_body(iol):
    clipped, truncated = iol._clip(b"hello")
    assert clipped == "hello"
    assert truncated is False


def test_parse_byte_size_suffixes(iol):
    assert iol._parse_byte_size("2MB", 0) == 2 * 1024 * 1024
    assert iol._parse_byte_size("1KB", 0) == 1024
    assert iol._parse_byte_size("1024", 0) == 1024
    assert iol._parse_byte_size("garbage", 99) == 99


def test_is_stream_detection(iol):
    assert iol._is_stream(b'{"stream": true}') is True
    assert iol._is_stream(b'{"stream": false}') is False
    assert iol._is_stream(b'{"messages": []}') is False
    assert iol._is_stream(b"not json") is False


def test_log_io_input_noop_when_disabled(iol, monkeypatch):
    # Must not raise and must short-circuit without env gate.
    monkeypatch.setenv("ENV_MODE", "prod")

    # log_io_input accepts a Request-like object; pass a minimal stand-in.
    class _FakeReq:
        method = "POST"
        url = None

    iol.log_io_input(_FakeReq(), b'{"stream":false}', "trace-1", "req-1")
