## ADDED Requirements

### Requirement: Snapshot installation dependency work is bounded
Cache-state SHALL validate and order a bounded worker snapshot in time and memory linear in the number of logical snapshot entries before atomically replacing worker ownership.

#### Scenario: Deep snapshot entries arrive in reverse dependency order
- **WHEN** a valid snapshot contains a long parent chain whose child entries appear before their ancestors
- **THEN** cache-state SHALL resolve every entry without repeatedly scanning the complete unresolved set

#### Scenario: Snapshot contains a cycle or missing parent
- **WHEN** dependency traversal cannot reach every snapshot entry from a root entry
- **THEN** cache-state SHALL reject the snapshot before changing existing HashTree ownership

#### Scenario: A block hash occurs in multiple chains
- **WHEN** distinct logical entries share a block hash but have different parent edges
- **THEN** cache-state SHALL retain each unique logical edge and SHALL allow any resolved occurrence of that block hash to satisfy child dependencies
