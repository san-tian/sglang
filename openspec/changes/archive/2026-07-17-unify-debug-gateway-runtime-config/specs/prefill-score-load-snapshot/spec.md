## MODIFIED Requirements

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
