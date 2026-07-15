# ttft-first-routing Specification

## Purpose
Define TTFT-first cache-aware routing semantics, including token-weighted local reservations and optional distributed cache-state matches.
## Requirements
### Requirement: TTFT pressure takes priority over cache affinity
When TTFT-first routing is enabled for cache-aware routing, the router SHALL select workers by the configured predicted first-token score mode before considering cache affinity. The default score mode SHALL retain the existing additive behavior; multiplicative score modes SHALL remain explicitly opt-in.

#### Scenario: Cache hit loses to lower predicted TTFT pressure
- **WHEN** a request has a prefix-cache hit on one worker and another healthy worker has a lower predicted first-token score outside the configured cache band
- **THEN** the router SHALL select the lower-score worker even though it has a weaker or missing cache hit

#### Scenario: Cache hit wins inside the TTFT pressure band
- **WHEN** multiple healthy workers have predicted first-token scores within the configured cache band
- **THEN** the router SHALL prefer the worker with the strongest prefix-cache match among those in-band workers

#### Scenario: Additive mode remains the default
- **WHEN** TTFT-first routing is enabled without an explicit score mode
- **THEN** the router SHALL use the existing additive request-pressure plus uncached-block score

### Requirement: Local reservations are token weighted
The router SHALL make local pending-load reservations proportional to the routed prompt token count for requests with derived routing tokens.

#### Scenario: Long prompt immediately affects later routing decisions
- **WHEN** a long prompt is assigned to a worker and the request is still in flight
- **THEN** subsequent TTFT-first selections SHALL see a larger local reservation on that worker before the next worker `/get_load` poll completes

#### Scenario: Missing tokens still reserves one request
- **WHEN** the router cannot derive routing tokens for a request
- **THEN** it SHALL still reserve at least one unit of local pending load for the selected worker

### Requirement: TTFT-first routing is opt-in
The router SHALL preserve existing cache-aware routing semantics unless the operator explicitly enables TTFT-first routing.

#### Scenario: Default cache-aware behavior remains unchanged
- **WHEN** cache-aware routing is configured without TTFT-first routing
- **THEN** worker selection SHALL continue to use the existing prefix-match, imbalance, and hit-load guard behavior

#### Scenario: TTFT-first flags require cache-aware routing
- **WHEN** an operator supplies TTFT-first routing flags with a policy other than `cache_aware_zmq`
- **THEN** router configuration validation SHALL fail at startup

### Requirement: TTFT-first routing can consume distributed cache matches
When TTFT-first routing is enabled with a remote cache-state service, the router SHALL compute each worker's predicted first-token pressure from remote prefix-match results, worker load, and local pending reservations.

#### Scenario: Remote cache hit lowers TTFT score
- **WHEN** a remote cache-state query reports matched blocks for one healthy worker
- **THEN** TTFT-first scoring SHALL reduce that worker's uncached prefill cost by the matched block count before comparing it with other workers

#### Scenario: Remote cache miss falls back to load pressure
- **WHEN** the remote cache-state query returns no matched workers
- **THEN** TTFT-first routing SHALL rank workers by load and pending pressure without cache affinity

### Requirement: Conservative LMetric multiplies prefill and batch pressure
In conservative LMetric mode, the router SHALL score a worker as the saturating product of candidate uncached tokens plus reported total waiting uncached tokens plus outstanding router-reserved prompt tokens, and one plus reported running requests plus outstanding router-reserved request count.

#### Scenario: Queued long prefill suppresses assignment
- **WHEN** two otherwise equivalent workers have different total waiting uncached token counts
- **THEN** conservative LMetric SHALL assign the lower score to the worker with less waiting uncached work

#### Scenario: Active batch amplifies prefill work
- **WHEN** two workers have equal prefill factors but different running-request counts
- **THEN** conservative LMetric SHALL assign the lower score to the worker with the smaller batch factor

### Requirement: Candidate-aware LMetric counts only work ahead
In candidate-aware LMetric mode, the router SHALL replace total waiting uncached work with chunked-prefill remainder, work at strictly better business priorities, and same-priority cumulative work in the candidate's uncached-length bucket.

#### Scenario: Bypassable long request does not suppress short candidate
- **WHEN** a queued long request is non-overdue and the worker summary indicates that a new same-priority short candidate would precede it
- **THEN** that long request's tokens SHALL not be included in the candidate's work-ahead factor

#### Scenario: Overdue request suppresses assignment
- **WHEN** a queued request is overdue and therefore precedes a new same-priority candidate
- **THEN** its uncached tokens SHALL be included in candidate work-ahead

#### Scenario: Better business priority suppresses assignment
- **WHEN** queued work has a strictly better business priority than the candidate
- **THEN** its uncached tokens SHALL be included regardless of length bucket

### Requirement: Multiplicative modes fail back compatibly
Multiplicative TTFT modes SHALL require TTFT-first routing and worker load polling. Every candidate in one selection SHALL be compared using the same score mode. Missing optional token data SHALL produce a deterministic pool-wide fallback, while an explicit load-probe failure SHALL retain existing unroutable or high-load handling.

#### Scenario: Candidate detail is incomplete
- **WHEN** candidate-aware mode receives conservative token totals for every candidate but any candidate lacks complete priority/bucket detail
- **THEN** the candidate set SHALL fall back together to conservative LMetric

#### Scenario: Old worker has no token fields
- **WHEN** any candidate's successful load response lacks the new prefill token fields
- **THEN** the candidate set SHALL fall back together to the existing additive TTFT score

#### Scenario: Native P/D candidates retain cache-aware routing
- **WHEN** a model is served by separately registered Prefill and Decode workers
- **THEN** the router SHALL continue applying prefix cache-aware selection to the Prefill candidate set but SHALL use additive scoring until paired Decode batch pressure is part of the score input

#### Scenario: Logical P/D proxy lacks comparable prefill state
- **WHEN** a logical proxy does not expose an explicit comparable Prefill queue snapshot and logical KV identity
- **THEN** the candidate set SHALL use existing additive cache-aware behavior rather than multiplicative scoring

#### Scenario: Load probe fails
- **WHEN** a worker load poll fails
- **THEN** the router SHALL preserve its existing failed-probe high-load or unroutable behavior

#### Scenario: Invalid flag combination
- **WHEN** an operator selects a multiplicative mode without TTFT-first routing or load polling, or combines it with idle-first prefiltering
- **THEN** router configuration validation SHALL fail at startup

### Requirement: Prefill-work-only mode separates PD phase scheduling
When Prefill-work-only mode is explicitly configured, the router SHALL score every compatible Prefill candidate as candidate uncached tokens plus reported total waiting uncached Prefill tokens plus outstanding Router-reserved prompt tokens, without multiplying by reported running requests or request-count reservations. Decode selection SHALL remain a separate downstream decision.

#### Scenario: Native PD Prefill uses token work
- **WHEN** every native Prefill candidate reports a compatible role-labelled Prefill token snapshot
- **THEN** the router SHALL compare those candidates using cache-adjusted Prefill token work and SHALL subsequently select Decode with the existing affinity and load policy

#### Scenario: Decode pressure is a constant for Prefill comparison
- **WHEN** native PD Prefill candidates report different running-request counts
- **THEN** Prefill-work-only scoring SHALL not use those counts as a Decode batch factor

#### Scenario: Router reservations prevent a stale Prefill hotspot
- **WHEN** a request has been assigned to a Prefill worker but its prompt tokens are not yet reflected by the next load poll
- **THEN** those reserved prompt tokens SHALL increase that worker's Prefill-work-only score

#### Scenario: Native PD snapshot is unavailable
- **WHEN** any Prefill candidate lacks compatible role-labelled Prefill token totals
- **THEN** the complete Prefill candidate set SHALL fall back to the existing additive TTFT score

#### Scenario: Logical PD proxy is not comparable
- **WHEN** a candidate is a logical PD proxy without guaranteed Prefill cache identity and Prefill token ownership
- **THEN** the candidate set SHALL use existing additive scoring rather than Prefill-work-only scoring

### Requirement: Logical PD proxy workers can receive group cache credit
When TTFT-first cache-aware routing evaluates a logical PD proxy candidate with configured physical Prefill members, the router SHALL treat a remote cache-state match on any configured member as cache-hit credit for that logical candidate.

#### Scenario: Member Prefill match credits logical worker
- **WHEN** a logical candidate is configured with physical Prefill members and remote cache-state returns one of those member URLs for the request prefix
- **THEN** TTFT-first scoring SHALL apply the matched block count to the logical candidate as if the logical candidate had a cache match

#### Scenario: Logical group score uses the best member score
- **WHEN** multiple physical worker candidates map to the same logical PD proxy group
- **THEN** TTFT-first routing SHALL compute that group's routing score from the best scoring member, including that member's cache match and load, and SHALL dispatch to the logical group worker URL rather than the physical member URL

#### Scenario: Direct worker match still works
- **WHEN** remote cache-state returns the candidate worker URL itself
- **THEN** TTFT-first scoring SHALL preserve the existing direct cache-hit behavior

#### Scenario: No member matches
- **WHEN** remote cache-state returns only physical worker URLs that are not the candidate URL and are not configured members of the candidate
- **THEN** TTFT-first scoring SHALL treat that candidate as having zero matched blocks

#### Scenario: No Prefill hint is sent
- **WHEN** the router selects a logical PD proxy candidate because one of its configured members has cache credit
- **THEN** the router SHALL dispatch the request to the logical worker URL without adding a preferred-Prefill header or otherwise forcing the inner PD Router's Prefill choice

#### Scenario: Logical group mapping is opt-in
- **WHEN** a logical PD proxy candidate is configured without physical Prefill members
- **THEN** TTFT-first scoring SHALL preserve the previous exact-worker-url cache match behavior
