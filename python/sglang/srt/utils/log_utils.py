from __future__ import annotations

import json
import logging
import os
import queue
import socket
import sys
import threading
import time
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional, Union

import torch.distributed as dist


def create_log_targets(
    *, targets: Optional[List[str]], name_prefix: str
) -> List[logging.Logger]:
    if not targets:
        return [_create_log_target_stdout(name_prefix)]
    return [_create_log_target(t, name_prefix) for t in targets]


def _create_log_target(target: str, name_prefix: str) -> logging.Logger:
    if target.lower() == "stdout":
        return _create_log_target_stdout(name_prefix)
    return _create_log_target_file(target, name_prefix)


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:
    return _create_logger_with_handler(
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(directory: str, name_prefix: str) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    filename = os.path.join(directory, f"{hostname}_{rank}.log")
    handler = TimedRotatingFileHandler(
        filename, when="H", backupCount=0, encoding="utf-8"
    )
    return _create_logger_with_handler(
        f"{name_prefix}.file.{directory}.{hostname}_{rank}", handler
    )


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def log_json(
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    msg = json.dumps(log_data, ensure_ascii=False)

    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger in loggers:
        logger.info(msg)


class SLSJsonFormatter(logging.Formatter):
    """Build one structured JSON object for stdout and Alibaba Cloud SLS."""

    def __init__(self, service_name: str = "sglang-worker"):
        super().__init__()
        self.service_name = service_name
        self.hostname = socket.gethostname()

    def build_dict(self, record: logging.LogRecord) -> dict:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "service_name": self.service_name,
            "hostname": self.hostname,
            "logger": record.name,
            "process": record.process,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["error"] = self.formatException(record.exc_info)
        for field in ("trace_id", "request_id", "rid", "model", "component"):
            value = getattr(record, field, None)
            if value:
                log_entry[field] = str(value)
        return log_entry

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self.build_dict(record), ensure_ascii=False)


class SLSLogContextFilter(logging.Filter):
    """Attach request-local trace identifiers to each log record."""

    def __init__(self):
        super().__init__()
        self._trace_id: ContextVar[Optional[str]] = ContextVar(
            "sls_trace_id", default=None
        )
        self._request_id: ContextVar[Optional[str]] = ContextVar(
            "sls_request_id", default=None
        )

    def set_context(
        self, trace_id: Optional[str] = None, request_id: Optional[str] = None
    ) -> tuple[Token, Token]:
        return self._trace_id.set(trace_id), self._request_id.set(request_id)

    def reset_context(self, tokens: tuple[Token, Token]) -> None:
        trace_token, request_token = tokens
        self._trace_id.reset(trace_token)
        self._request_id.reset(request_token)

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = self._trace_id.get()
        request_id = self._request_id.get()
        if trace_id and not hasattr(record, "trace_id"):
            record.trace_id = trace_id
        if request_id and not hasattr(record, "request_id"):
            record.request_id = request_id
        return True


_global_sls_filter = SLSLogContextFilter()


def get_sls_log_filter() -> SLSLogContextFilter:
    return _global_sls_filter


def _bool_env(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("true", "1", "yes", "on")


class SLSLogHandler(logging.Handler):
    """Bounded, fail-open background batch writer for Alibaba Cloud SLS."""

    QUEUE_CAPACITY = 10_000
    BATCH_SIZE = 200
    FLUSH_INTERVAL_SECS = 1.0
    HTTP_TIMEOUT_SECS = 5

    def __init__(self, service_name: str = "sglang-worker"):
        super().__init__()
        self.service_name = service_name
        self.formatter = SLSJsonFormatter(service_name=service_name)
        self._queue: queue.Queue[Optional[dict]] = queue.Queue(
            maxsize=self.QUEUE_CAPACITY
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self._project = ""
        self._logstore = ""
        self._sls_closed = False
        self.dropped_queue_full = 0
        self.dropped_send_error = 0
        self.sent_entries = 0
        self._last_send_error_report_at = 0.0

        self._init_client()
        if self._client is not None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="sls-log-pusher"
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _init_client(self) -> None:
        endpoint = os.environ.get("SLS_ENDPOINT", "").strip()
        access_key_id = os.environ.get("SLS_ACCESS_KEY_ID", "").strip()
        access_key_secret = os.environ.get("SLS_ACCESS_KEY_SECRET", "").strip()
        self._project = os.environ.get("SLS_PROJECT", "macaron-log").strip()
        self._logstore = os.environ.get("SLS_LOGSTORE", "sglang-worker").strip()
        if not endpoint or not access_key_id or not access_key_secret:
            return

        try:
            from aliyun.log import LogClient
        except ImportError:
            return

        self._client = LogClient(endpoint, access_key_id, access_key_secret)
        # The SDK default is 120 seconds, which is unsafe for a logging thread
        # that must terminate promptly during a rolling restart.
        self._client.timeout = self.HTTP_TIMEOUT_SECS

    def emit(self, record: logging.LogRecord) -> None:
        if self._client is None or self._stop.is_set():
            return
        try:
            self._queue.put_nowait(self.formatter.build_dict(record))
        except queue.Full:
            self.dropped_queue_full += 1
        except Exception:
            self.handleError(record)

    def _run(self) -> None:
        from aliyun.log import LogItem, PutLogsRequest

        batch: list[dict] = []
        flush_deadline: Optional[float] = None
        while True:
            if self._stop.is_set() and self._queue.empty():
                break

            timeout = self.FLUSH_INTERVAL_SECS
            if flush_deadline is not None:
                timeout = max(0.0, flush_deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._flush(batch, LogItem, PutLogsRequest)
                batch = []
                flush_deadline = None
                continue

            if item is None:
                break
            if not batch:
                flush_deadline = time.monotonic() + self.FLUSH_INTERVAL_SECS
            batch.append(item)
            if len(batch) >= self.BATCH_SIZE:
                self._flush(batch, LogItem, PutLogsRequest)
                batch = []
                flush_deadline = None

        self._drain_and_flush(batch, LogItem, PutLogsRequest)

    def _drain_and_flush(self, batch: list[dict], LogItem, PutLogsRequest) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            batch.append(item)
            if len(batch) >= self.BATCH_SIZE:
                self._flush(batch, LogItem, PutLogsRequest)
                batch = []
        self._flush(batch, LogItem, PutLogsRequest)

    def _flush(self, batch: list[dict], LogItem, PutLogsRequest) -> None:
        if not batch or self._client is None:
            return
        timestamp = int(time.time())
        items = [
            LogItem(
                timestamp=timestamp,
                contents=[(key, str(value)) for key, value in log_dict.items()],
            )
            for log_dict in batch
        ]
        try:
            request = PutLogsRequest(
                self._project, self._logstore, topic="", logitems=items
            )
            self._client.put_logs(request)
            self.sent_entries += len(batch)
        except Exception as error:
            self.dropped_send_error += len(batch)
            now = time.monotonic()
            if now - self._last_send_error_report_at >= 60:
                self._last_send_error_report_at = now
                print(
                    "SLS batch delivery failed; dropping batch "
                    f"({type(error).__name__})",
                    file=sys.stderr,
                )

    def close(self) -> None:
        if self._sls_closed:
            return
        self._sls_closed = True
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # The stop flag makes the worker drain the full queue without a
            # sentinel, while keeping close() non-blocking on the caller.
            pass
        if self._thread is not None:
            self._thread.join(timeout=self.HTTP_TIMEOUT_SECS + 1)
        super().close()


def configure_sls_logging(service_name: str = "sglang-worker") -> None:
    """Enable structured stdout and optional direct SLS delivery."""
    if not _bool_env("SGLANG_SLS_LOGGING"):
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        if isinstance(handler, SLSLogHandler):
            handler.close()

    formatter = SLSJsonFormatter(service_name=service_name)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(_global_sls_filter)
    root_logger.addHandler(stdout_handler)

    sls_handler = SLSLogHandler(service_name=service_name)
    if sls_handler.enabled:
        sls_handler.addFilter(_global_sls_filter)
        root_logger.addHandler(sls_handler)
    else:
        sls_handler.close()
        print(
            "SLS logging enabled but credentials or aliyun-log-python-sdk are unavailable; using stdout only",
            file=sys.stderr,
        )
