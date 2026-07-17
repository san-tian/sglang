## MODIFIED Requirements

### Requirement: Gateway can use remote cache state
The gateway SHALL support an opt-in remote cache-state client for cache-aware routing. In `remote_only` mode the remote response SHALL be the only cache-match source: useful trusted matches SHALL participate in worker scoring, while authoritative misses, transport failures, timeouts, and malformed responses SHALL continue through cache-miss Prefill/load selection without consulting or populating a local route-history tree. Other explicitly configured cache-tree modes SHALL preserve their existing behavior.

#### Scenario: Remote cache-state URL is configured
- **WHEN** cache-aware routing is enabled in `remote_only` mode with a remote cache-state URL and the remote service returns a useful trusted prefix match
- **THEN** the gateway SHALL use the remote match result for prefix-aware worker scoring

#### Scenario: Authoritative remote service returns no useful trusted match
- **WHEN** the remote query succeeds with an authoritative response but returns zero matched blocks or no trusted matching workers
- **THEN** the gateway SHALL skip local cache-history matching and continue through cache-miss load-based selection

#### Scenario: Remote-only service is unavailable or malformed
- **WHEN** the remote query fails, times out, or returns malformed data in `remote_only` mode
- **THEN** the gateway SHALL treat cache match as empty, SHALL NOT query a local prefix tree, and SHALL continue through Prefill/load fallback

#### Scenario: Another cache-tree mode is explicitly configured
- **WHEN** the administrator selects an existing non-remote-only source
- **THEN** the gateway SHALL preserve that source's existing local and remote fallback semantics
