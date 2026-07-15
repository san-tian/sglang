## ADDED Requirements

### Requirement: Cache-state preserves physical Prefill ownership for logical groups
Cache-state SHALL continue to store and return KV prefix ownership using physical worker URLs and DP ranks, even when those physical workers are members of a logical PD proxy worker.

#### Scenario: Physical Prefill member matches
- **WHEN** a physical Prefill member has published KV events for a prefix and a gateway queries cache-state for that prefix
- **THEN** cache-state SHALL return the physical Prefill worker URL and DP rank rather than a synthetic logical proxy URL

#### Scenario: Multiple Prefill members exist in one logical group
- **WHEN** two physical Prefill members belong to the same logical proxy group and only one member holds the deepest prefix
- **THEN** cache-state SHALL report only the physical member or members that actually hold that prefix

#### Scenario: Logical group mapping is absent
- **WHEN** a gateway has no logical group mapping for a returned physical Prefill worker URL
- **THEN** cache-state SHALL NOT synthesize any logical worker match on the gateway's behalf
