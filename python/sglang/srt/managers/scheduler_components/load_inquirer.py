from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    PrefillQueueMetrics,
    PrefillWorkMetrics,
    PrefillWorkOverflowMetrics,
    PrefillWorkRequestMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.managers.schedule_policy import (
    PREFILL_QUEUE_PRIORITY_GROUP_LIMIT,
    PREFILL_WORK_BUCKET_BOUNDS,
    prefill_one_oldest_three_shortest_order,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)

PREFILL_WORK_SCHEMA_VERSION = 1
PREFILL_WORK_DETAIL_LIMIT = 64


def _length_bucket(tokens: int) -> int:
    for bound in PREFILL_WORK_BUCKET_BOUNDS:
        if tokens <= bound:
            return bound
    return PREFILL_WORK_BUCKET_BOUNDS[-1]


def _uncached_tokens(req) -> int:
    return max(0, req.seqlen - req.num_matched_prefix_tokens)


def _priority(req, enabled: bool) -> int:
    return req.priority if enabled and req.priority is not None else 0


def _build_prefill_work_metrics(
    waiting_queues,
    running_reqs,
    chunked_req,
    *,
    priority_scheduling_enabled: bool,
    schedule_low_priority_values_first: bool,
) -> PrefillWorkMetrics:
    """Build a bounded request-level snapshot without changing scheduling."""
    waiting = []
    running = []
    overflow = {}
    truncated = False

    def add_overflow(req, tokens: int):
        key = (_priority(req, priority_scheduling_enabled), _length_bucket(tokens))
        count, total = overflow.get(key, (0, 0))
        overflow[key] = (count + 1, total + tokens)

    for queue in waiting_queues:
        for req in queue:
            tokens = _uncached_tokens(req)
            if len(waiting) < PREFILL_WORK_DETAIL_LIMIT:
                waiting.append(
                    PrefillWorkRequestMetrics(
                        request_id=str(req.rid),
                        priority=_priority(req, priority_scheduling_enabled),
                        total_uncached_tokens=tokens,
                        processed_uncached_tokens=0,
                        current_chunk_end_tokens=tokens,
                    )
                )
            else:
                truncated = True
                add_overflow(req, tokens)

    running_candidates = list(running_reqs)
    if chunked_req is not None and all(
        req is not chunked_req for req in running_candidates
    ):
        running_candidates.append(chunked_req)

    for req in running_candidates:
        total = _uncached_tokens(req)
        cached_prefix = max(0, req.num_matched_prefix_tokens)
        processed = min(
            total,
            max(0, len(req.prefix_indices) - cached_prefix),
        )
        if processed >= total:
            continue
        chunk_end = total
        if req is chunked_req and req.extend_range is not None:
            chunk_end = min(
                total,
                max(processed, req.extend_range.end - cached_prefix),
            )
        entry = PrefillWorkRequestMetrics(
            request_id=str(req.rid),
            priority=_priority(req, priority_scheduling_enabled),
            total_uncached_tokens=total,
            processed_uncached_tokens=processed,
            current_chunk_end_tokens=chunk_end,
        )
        if len(running) < PREFILL_WORK_DETAIL_LIMIT:
            running.append(entry)
        else:
            truncated = True
            add_overflow(req, max(0, total - processed))

    overflow_entries = tuple(
        PrefillWorkOverflowMetrics(
            priority=priority,
            length_bucket=bucket,
            request_count=count,
            total_uncached_tokens=total,
        )
        for (priority, bucket), (count, total) in sorted(overflow.items())
    )
    now_ms = int(time.time() * 1000)
    return PrefillWorkMetrics(
        schema_version=PREFILL_WORK_SCHEMA_VERSION,
        snapshot_id=time.time_ns(),
        generated_at_ms=now_ms,
        worker_boot_id=os.environ.get("SGLANG_WORKER_BOOT_ID", str(os.getpid())),
        priority_scheduling_enabled=priority_scheduling_enabled,
        schedule_low_priority_values_first=schedule_low_priority_values_first,
        detail_complete=not truncated,
        truncated=truncated,
        waiting_prefill=tuple(waiting),
        running_prefill=tuple(running),
        overflow_summary=overflow_entries,
    )


def build_prefill_queue_metrics(
    waiting_queue,
    chunked_req,
    *,
    now: float,
    aging_rate: float,
    max_wait_seconds: float,
    priority_scheduling_enabled: bool,
    schedule_low_priority_values_first: bool,
) -> PrefillQueueMetrics:
    """Compress waiting work into bounded priority and candidate-work buckets."""
    groups = {}
    detail_complete = True
    for req in waiting_queue:
        priority = (
            req.priority
            if priority_scheduling_enabled and req.priority is not None
            else 0
        )
        if priority not in groups:
            if len(groups) >= PREFILL_QUEUE_PRIORITY_GROUP_LIMIT:
                detail_complete = False
                groups.clear()
                break
            groups[priority] = [0, [], [0] * len(PREFILL_WORK_BUCKET_BOUNDS)]

        uncached_tokens = max(0, req.seqlen - req.num_matched_prefix_tokens)
        group = groups[priority]
        group[0] += uncached_tokens
        group[1].append(
            (
                uncached_tokens,
                req.time_stats.wait_queue_entry_time,
                len(group[1]),
            )
        )

    if detail_complete:
        for group in groups.values():
            states = group[1]
            for bucket_index, upper_bound in enumerate(PREFILL_WORK_BUCKET_BOUNDS):
                candidate_index = len(states)
                candidate_state = (upper_bound, float("inf"), candidate_index)
                order = prefill_one_oldest_three_shortest_order(
                    states + [candidate_state]
                )
                group[2][bucket_index] = sum(
                    states[index][0] for index in order[: order.index(candidate_index)]
                )

    priority_values = tuple(sorted(groups)) if detail_complete else ()
    chunked_remaining = (
        max(0, chunked_req.seqlen - len(chunked_req.prefix_indices))
        if chunked_req is not None
        else 0
    )
    return PrefillQueueMetrics(
        detail_complete=detail_complete,
        chunked_remaining_uncached_tokens=chunked_remaining,
        work_bucket_bounds=PREFILL_WORK_BUCKET_BOUNDS,
        priority_scheduling_enabled=priority_scheduling_enabled,
        schedule_low_priority_values_first=schedule_low_priority_values_first,
        priority_values=priority_values,
        priority_total_uncached_tokens=tuple(groups[p][0] for p in priority_values),
        priority_ahead_uncached_tokens=tuple(
            tuple(groups[p][2]) for p in priority_values
        ),
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    disaggregation_mode: DisaggregationMode
    ps: ParallelState
    server_args: ServerArgs
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    get_running_batch: Callable
    get_waiting_queue: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable
    get_spec_total_num_accept_tokens: Callable
    get_spec_total_num_forward_ct: Callable

    def _get_num_pending_tokens(self, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.get_waiting_queue())
        if self.get_chunked_req() is not None:
            req = self.get_chunked_req()
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_num_waiting_uncached_tokens(self) -> int:
        """Get uncached input tokens waiting for prefill compute."""
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return 0

        waiting_queues = [self.get_waiting_queue()]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.get_disagg_prefill_bootstrap_queue().queue)

        num_tokens = 0
        for queue in waiting_queues:
            for req in queue:
                num_tokens += max(0, req.seqlen - req.num_matched_prefix_tokens)
        cr = self.get_chunked_req()
        if cr is not None:
            num_tokens += max(0, cr.seqlen - len(cr.prefix_indices))
        return num_tokens

    def get_loads(self, req: GetLoadsReqInput = None) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections
        """
        if req is None:
            req = GetLoadsReqInput()

        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(self.get_running_batch().reqs)

        waiting_queues = [self.get_waiting_queue()]
        pending_token_queues = [self.get_waiting_queue()]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            prefill_bootstrap_queue = self.get_disagg_prefill_bootstrap_queue().queue
            waiting_queues.append(prefill_bootstrap_queue)
            pending_token_queues.append(prefill_bootstrap_queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            decode_prealloc_queue = self.get_disagg_decode_prealloc_queue().queue
            decode_transfer_queue = self.get_disagg_decode_transfer_queue().queue
            decode_retracted_queue = (
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
            waiting_queues.append(decode_prealloc_queue)
            waiting_queues.append(decode_transfer_queue)
            waiting_queues.append(decode_retracted_queue)
            # In disaggregated decode, transfer-queue requests and transferred
            # waiting-queue requests have already pre-allocated decode-side KV
            # slots, so they are already included in num_used_tokens.
            pending_token_queues = [decode_prealloc_queue, decode_retracted_queue]

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_waiting_uncached_tokens = self.get_num_waiting_uncached_tokens()
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in pending_token_queues for req in queue
        )

        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if include_all or "spec" in include:
            if (
                not self.spec_algorithm.is_none()
                and self.get_spec_total_num_forward_ct() > 0
            ):
                speculative = SpeculativeMetrics(
                    accept_length=(
                        self.get_spec_total_num_accept_tokens()
                        / self.get_spec_total_num_forward_ct()
                    ),
                    accept_rate=self.get_stats().spec_accept_rate,
                )

        lora = None
        if include_all or "lora" in include:
            if self.server_args.enable_lora:
                lora = LoRAMetrics(
                    slots_used=self.get_stats().lora_pool_slots_used,
                    slots_total=self.get_stats().lora_pool_slots_total,
                    utilization=self.get_stats().lora_pool_utilization,
                )

        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_bootstrap = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_bootstrap = len(self.get_disagg_prefill_bootstrap_queue().queue)
                prefill_inflight = len(self.get_disagg_prefill_inflight_queue())
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(self.get_disagg_decode_prealloc_queue().queue)
                decode_transfer = len(self.get_disagg_decode_transfer_queue().queue)
                decode_retracted = len(
                    self.get_disagg_decode_prealloc_queue().retracted_queue
                )

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_bootstrap_queue_reqs=prefill_bootstrap,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=self.get_stats().kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=self.get_stats().kv_transfer_latency_ms,
            )

        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(self.get_waiting_queue()),
                grammar=self.get_stats().num_grammar_queue_reqs,
                paused=self.get_stats().num_paused_reqs,
                retracted=self.get_stats().num_retracted_reqs,
            )

        prefill_queue = None
        if (
            (include_all or "prefill_queue" in include)
            and self.server_args.schedule_policy == "prefill-length-aware"
            and self.disaggregation_mode != DisaggregationMode.DECODE
        ):
            prefill_queue = build_prefill_queue_metrics(
                self.get_waiting_queue(),
                self.get_chunked_req(),
                now=time.perf_counter(),
                aging_rate=self.server_args.prefill_length_aware_aging_rate,
                max_wait_seconds=self.server_args.prefill_length_aware_max_wait_seconds,
                priority_scheduling_enabled=self.server_args.enable_priority_scheduling,
                schedule_low_priority_values_first=(
                    self.server_args.schedule_low_priority_values_first
                ),
            )

        prefill_work = None
        if (
            include_all or "prefill_work" in include
        ) and self.disaggregation_mode != DisaggregationMode.DECODE:
            prefill_work = _build_prefill_work_metrics(
                waiting_queues,
                self.get_running_batch().reqs,
                self.get_chunked_req(),
                priority_scheduling_enabled=self.server_args.enable_priority_scheduling,
                schedule_low_priority_values_first=self.server_args.schedule_low_priority_values_first,
            )

        return GetLoadsReqOutput(
            dp_rank=self.ps.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=num_waiting_uncached_tokens,
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(self.get_stats().gen_throughput, 2),
            cache_hit_rate=round(self.get_stats().cache_hit_rate, 4),
            utilization=round(self.get_stats().utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
            prefill_queue=prefill_queue,
            prefill_work=prefill_work,
        )
