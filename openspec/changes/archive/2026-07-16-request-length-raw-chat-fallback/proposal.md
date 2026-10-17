# Opt-in raw prompt context routing

## Why

Some deployed chat models do not expose a tokenizer chat template to the gateway. With context-bounded workers, those requests are conservatively classified as unknown and rejected even when the raw message text can be tokenized. The isolated length-split debug gateway needs an explicit opt-in to exercise short AMD and long NVIDIA routing for this model.

## What Changes

- Add an opt-in `ALLOW_RAW_CONTEXT_TOKENS` setting.
- When enabled, generation routes may use the router's raw prompt tokenization for context-range eligibility when no engine-equivalent chat encoder is available.
- Keep the default fail-closed behavior and never enable the option implicitly in production configurations.
- Add unit and HTTP coverage for opt-in raw chat routing and default fail-closed behavior.

## Scope

This change affects router code and tests only. The isolated debug deployment enables the setting; production gateways remain unchanged.
