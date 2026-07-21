"""Full upstream request/response body logging for troubleshooting.

Mirrors the relay (macaron-relay-stack, Go) `UPSTREAM_IO_LOG` and the sgl-router
(Rust) `io_log` designs: when enabled, the sgl-worker prints the full LLM request
input and response output so agents and operators can troubleshoot agent-loop
problems end to end. This is the worker-side counterpart — together the three
services cover 中转站 → SGLang Router → Worker.

Gated by `UPSTREAM_IO_LOG=1` or `ENV_MODE=alpha` (alpha defaults on for integration
testing). Disabled otherwise, so production traffic is unaffected. The logged copy
is capped at ``IO_LOG_MAX_BODY_BYTES`` so a single long SSE never blows up the log
volume; the live client stream is never truncated.
"""

from __future__ import annotations

import json
import logging
import os

from starlette.requests import Request
from starlette.responses import Response

# Per-side logged-copy cap default. Override via UPSTREAM_IO_LOG_MAX_BODY_BYTES
# (plain bytes or KB/MB/GB suffix, e.g. 2MB). The live client stream is never
# capped; only the logged copy.
IO_LOG_MAX_BODY_BYTES_DEFAULT = 2 * 1024 * 1024
_UPSTREAM_IO_LOG_ENV = "UPSTREAM_IO_LOG"
_ENV_MODE_ENV = "ENV_MODE"
_MAX_BODY_BYTES_ENV = "UPSTREAM_IO_LOG_MAX_BODY_BYTES"

logger = logging.getLogger("sglang.srt.io_log")


def io_log_enabled() -> bool:
    """Report whether full upstream input/output logging is on."""
    if os.environ.get(_UPSTREAM_IO_LOG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    ):
        return True
    return os.environ.get(_ENV_MODE_ENV, "").strip().lower() == "alpha"


def _max_body_bytes() -> int:
    """Resolve the per-side logged-copy cap from the env var, default 2MB."""
    return _parse_byte_size(
        os.environ.get(_MAX_BODY_BYTES_ENV, ""), IO_LOG_MAX_BODY_BYTES_DEFAULT
    )


def _parse_byte_size(raw: str, default: int) -> int:
    """Parse a byte count with an optional KB/MB/GB suffix. Falls back to default."""
    raw = raw.strip()
    if not raw:
        return default
    lower = raw.lower()
    mult = 1
    if lower.endswith("gb"):
        mult, num = 1 << 30, raw[:-2]
    elif lower.endswith("mb"):
        mult, num = 1 << 20, raw[:-2]
    elif lower.endswith("kb"):
        mult, num = 1 << 10, raw[:-2]
    else:
        num = raw
    try:
        n = int(num.strip())
    except ValueError:
        return default
    if n < 0:
        return default
    return n * mult


def _clip(body: bytes) -> tuple[str, bool]:
    """Clip to the configured cap, keeping head + tail and dropping the middle when
    truncated so both the prompt opening and the finish/[DONE] stay visible."""
    cap = _max_body_bytes()
    if len(body) <= cap:
        return body.decode("utf-8", errors="replace"), False
    half = cap // 2
    head = body[:half].decode("utf-8", errors="replace")
    tail = body[len(body) - half :].decode("utf-8", errors="replace")
    dropped = len(body) - cap
    return f"{head}\n...[truncated {dropped} bytes]...\n{tail}", True


def _target(request: Request) -> str:
    if request.url is not None:
        return str(request.url)
    return ""


def log_io_input(request: Request, body: bytes, trace_id: str, request_id: str) -> None:
    if not io_log_enabled():
        return
    clipped, truncated = _clip(body)
    logger.info(
        "upstream_io_input",
        extra={
            "io_event": "upstream_io_input",
            "trace_id": trace_id,
            "x_request_id": request_id,
            "method": request.method,
            "path": request.url.path if request.url else "",
            "target": _target(request),
            "stream": _is_stream(body),
            "body_truncated": truncated,
            "request_body": clipped,
        },
    )


def log_io_output(
    response: Response,
    body: bytes,
    trace_id: str,
    request_id: str,
    method: str,
    path: str,
    target: str,
    stream: bool,
) -> None:
    if not io_log_enabled():
        return
    clipped, truncated = _clip(body)
    logger.info(
        "upstream_io_output",
        extra={
            "io_event": "upstream_io_output",
            "trace_id": trace_id,
            "x_request_id": request_id,
            "method": method,
            "path": path,
            "target": target,
            "stream": stream,
            "status_code": getattr(response, "status_code", None),
            "body_truncated": truncated,
            "response_body": clipped,
        },
    )


def _is_stream(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return False
    return bool(isinstance(payload, dict) and payload.get("stream"))


def log_io_output_clipped(
    response,
    body: str,
    truncated: bool,
    trace_id: str,
    request_id: str,
    method: str,
    path: str,
    target: str,
    stream: bool,
) -> None:
    """Log a streaming response whose head+tail has already been clipped by the tee.

    Unlike ``log_io_output`` this takes the already-materialized (head+tail) string and
    truncation flag, so the streaming tee can keep both the opening chunks and the
    finish/[DONE] without buffering the whole response.
    """
    if not io_log_enabled():
        return
    logger.info(
        "upstream_io_output",
        extra={
            "io_event": "upstream_io_output",
            "trace_id": trace_id,
            "x_request_id": request_id,
            "method": method,
            "path": path,
            "target": target,
            "stream": stream,
            "status_code": getattr(response, "status_code", None),
            "body_truncated": truncated,
            "response_body": body,
        },
    )


__all__ = [
    "IO_LOG_MAX_BODY_BYTES_DEFAULT",
    "io_log_enabled",
    "log_io_input",
    "log_io_output",
    "log_io_output_clipped",
]
