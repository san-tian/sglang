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
from starlette.types import Message

# Per-side logged-copy cap. Raised from 64KB to 2MB because a glm-5.2 reasoning
# response (reasoning_content + final content) routinely exceeds 64KB and was
# being truncated, making the logged response_body look incomplete. The live
# client stream is never capped; only the logged copy is.
IO_LOG_MAX_BODY_BYTES = 2 * 1024 * 1024
_UPSTREAM_IO_LOG_ENV = "UPSTREAM_IO_LOG"
_ENV_MODE_ENV = "ENV_MODE"

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


def _clip(body: bytes) -> tuple[str, bool]:
    truncated = len(body) > IO_LOG_MAX_BODY_BYTES
    end = IO_LOG_MAX_BODY_BYTES if truncated else len(body)
    return body[:end].decode("utf-8", errors="replace"), truncated


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


__all__ = [
    "IO_LOG_MAX_BODY_BYTES",
    "io_log_enabled",
    "log_io_input",
    "log_io_output",
]

