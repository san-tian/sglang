# prefill-score-load-snapshot Specification

## Purpose
TBD - created by archiving change gateway-prefill-score-fallback. Update Purpose after archive.
## Requirements
### Requirement: Worker exposes a bounded versioned Prefill load snapshot
The SGLang worker SHALL expose an optional `/v1/loads` Prefill work section containing a schema version, snapshot identifier, generation timestamp, worker boot identifier, detail completeness, truncation state, and bounded waiting/running Prefill entries.

#### Scenario: Fresh complete snapshot
- **WHEN** the worker has a valid scheduler snapshot and the requested detail section is enabled
- **THEN** the response includes the current snapshot metadata and bounded waiting/running Prefill entries with uncached token information

#### Scenario: Snapshot exceeds the response bound
- **WHEN** the waiting or running entry count exceeds the configured response limit
- **THEN** the worker sets `truncated=true` and returns aggregate overflow information instead of an unbounded request list

### Requirement: Running Prefill entries expose remaining work boundaries
Each running Prefill entry SHALL expose a stable internal request identifier, priority, total uncached tokens, processed uncached tokens, and the current chunk end needed to calculate remaining Prefill work.

#### Scenario: Chunked Prefill is in progress
- **WHEN** a request has processed part of its uncached input and has a scheduler chunk boundary
- **THEN** the response allows the Gateway to calculate `F(current_chunk_end) - F(processed_tokens)` for that request

### Requirement: Gateway computes Prefill-only routing scores
The Gateway SHALL compute candidate scores in Prefill time units using waiting work, running remaining work, and the new request's effective Prefill time; PD candidates SHALL support a length-bucket speed coefficient without adding Decode or communication load to the score. Calibrated `curve_ms` and `pd_beta` profiles SHALL be read from `PREFILL_SCORE_PROFILES_JSON` when configured.

#### Scenario: Integrated worker candidate
- **WHEN** a complete or compatible Prefill load snapshot is available for an integrated worker
- **THEN** the score includes the Prefill curve for work ahead and the candidate's uncached input, with no Decode term

#### Scenario: PD worker candidate
- **WHEN** a PD Prefill candidate has a configured speed coefficient for the input length bucket
- **THEN** the Gateway uses `G(x) = F(x) / beta(x)` for both queue work and new-request work, while existing health/routability checks remain mandatory

#### Scenario: Environment profile is absent
- **WHEN** the environment contains no calibrated Prefill profile
- **THEN** the Gateway SHALL use its existing fixed-throughput fallback and SHALL NOT invent hardware-specific curve values

### Requirement: Load polling is asynchronous and provenance is retained
The Gateway SHALL poll worker load snapshots in the background, use a last-good snapshot in the routing path, and record the source, age, schema, truncation, and fallback reason for each decision. The administrator runtime configuration SHALL set the stale grace, consecutive failure threshold, and consecutive recovery threshold used by this state machine.

#### Scenario: Load endpoint is temporarily unreachable
- **WHEN** `/v1/loads` times out but the worker health and routing probe remain healthy and the last-good snapshot is within the configured stale grace
- **THEN** the Gateway SHALL use the stale snapshot with conservative age handling instead of synchronously waiting or treating the worker as zero load

#### Scenario: Consecutive failure threshold is reached
- **WHEN** load polling fails for the configured number of consecutive attempts
- **THEN** the Gateway SHALL stop treating the snapshot as fresh and SHALL select the next ordered reservation/fixed-capacity fallback

#### Scenario: Consecutive recovery threshold is reached
- **WHEN** the load endpoint succeeds for the configured number of consecutive attempts after failure
- **THEN** the Gateway SHALL restore snapshot use with fresh provenance

#### Scenario: Worker health fails
- **WHEN** the worker health or routing probe fails
- **THEN** the worker SHALL be fail-closed and SHALL NOT be selected regardless of its last load snapshot

### Requirement: Gateway provides unit-safe load fallbacks
The Gateway SHALL provide ordered fallbacks from enhanced request-level snapshot to compatible aggregate fields, stale snapshot plus reservation, shared reservation, local reservation, and fixed-capacity scoring, converting every usable source to comparable Prefill time units.

#### Scenario: Enhanced schema is unavailable
- **WHEN** the worker returns an older schema or only aggregate waiting uncached tokens
- **THEN** the Gateway uses the compatible aggregate fallback and does not apply a nonlinear curve to the aggregate token count as if it were one request

#### Scenario: All load state is unavailable
- **WHEN** the worker snapshot and shared reservation cannot be read
- **THEN** the Gateway uses local reservation or pool-wide fixed-capacity scoring and records the fallback reason; it never interprets the missing load as zero

### Requirement: Overlapping worker snapshot and reservation work is not double-counted
The Gateway SHALL use the shared internal request identifier for exact deduplication when available and a conservative same-unit maximum when legacy sources cannot be deduplicated.

#### Scenario: Request exists in both sources
- **WHEN** a request identifier appears in both the worker snapshot and Router reservation
- **THEN** that request contributes only once to the candidate's Prefill work

#### Scenario: Legacy source has no request identifiers
- **WHEN** a legacy aggregate snapshot overlaps Router reservation without per-request identifiers
- **THEN** the Gateway uses the maximum of the two same-unit work estimates instead of summing both full estimates
