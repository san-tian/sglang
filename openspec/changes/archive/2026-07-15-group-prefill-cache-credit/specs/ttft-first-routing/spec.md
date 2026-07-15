## ADDED Requirements

### Requirement: Logical PD proxy workers can receive group cache credit
When TTFT-first cache-aware routing evaluates a logical PD proxy candidate with configured physical Prefill members, the router SHALL treat a remote cache-state match on any configured member as cache-hit credit for that logical candidate.

#### Scenario: Member Prefill match credits logical worker
- **WHEN** a logical candidate is configured with physical Prefill members and remote cache-state returns one of those member URLs for the request prefix
- **THEN** TTFT-first scoring SHALL apply the matched block count to the logical candidate as if the logical candidate had a cache match

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
