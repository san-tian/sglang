# gateway-runtime-configuration Specification

## Purpose
TBD - created by archiving change unify-debug-gateway-runtime-config. Update Purpose after archive.
## Requirements
### Requirement: Administrator YAML declares the debug Gateway runtime contract
The Gateway SHALL accept a strict runtime configuration that selects remote-only cache state, Prefill snapshot fallback thresholds, an environment-backed optional Prefill profile, required Redis RouterState, required SLS logging, and route-decision logging.

#### Scenario: Complete runtime configuration
- **WHEN** an administrator configuration declares all supported runtime sections and required environment dependencies are present
- **THEN** the Gateway SHALL compile them into the existing runtime contract before worker discovery or listener creation

#### Scenario: Unknown runtime field
- **WHEN** the runtime configuration contains an unknown field or unsupported mode
- **THEN** startup validation SHALL fail instead of silently ignoring configuration drift

### Requirement: Required infrastructure dependencies fail closed at startup
The Gateway SHALL validate the fixed cache-state, Redis RouterState, and SLS environment contracts when their runtime modes are required, without logging their values.

#### Scenario: Required dependency is missing
- **WHEN** remote-only cache state, Redis RouterState, or SLS is required and one of its mandatory environment variables is empty or absent
- **THEN** validation SHALL fail before the Gateway listens for traffic and SHALL identify only the missing variable name

#### Scenario: Required secrets are present
- **WHEN** all required variables are non-empty
- **THEN** validation SHALL succeed without serializing or logging any secret value

### Requirement: Prefill profiles remain environment-owned and optional
The Gateway SHALL read `curve_ms` and `pd_beta` only from `PREFILL_SCORE_PROFILES_JSON` when the runtime profile source is `environment_optional`, and SHALL preserve fixed-throughput fallback when that variable is absent.

#### Scenario: Calibrated profile is configured
- **WHEN** `PREFILL_SCORE_PROFILES_JSON` contains valid per-selector curves and PD coefficients
- **THEN** PrefillScore SHALL use the configured curves through the existing validated profile parser

#### Scenario: Profile is absent
- **WHEN** no profile JSON is configured
- **THEN** Gateway startup SHALL remain valid and scoring SHALL record fixed-throughput fallback provenance

### Requirement: Administrator key inventory is exactly four keys
The administrator configuration SHALL expose only `internal-fp8-low`, `external-amd-high`, `external-all-length-high`, and `external-nvidia-high`, with their configured model, priority, hardware, quantization, and length-routing scopes.

#### Scenario: Legacy debug key is presented
- **WHEN** a client presents a credential that is not one of the four configured keys
- **THEN** the Gateway SHALL return the existing sanitized authentication failure and SHALL NOT route the request

### Requirement: Candidate source contains live source history
The release gate SHALL verify each identified live debug Gateway and SGLang worker source commit is an ancestor of the candidate feature branch or is explicitly merged before candidate build.

#### Scenario: Live commit is already included
- **WHEN** the live commit is an ancestor of the candidate commit
- **THEN** the release audit SHALL record it as included without duplicating its changes

#### Scenario: Live commit is missing
- **WHEN** an identified live commit is not an ancestor of the candidate
- **THEN** candidate deployment SHALL remain blocked until the difference is reviewed and merged or explicitly excluded with evidence

