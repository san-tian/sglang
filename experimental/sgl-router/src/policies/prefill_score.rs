// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use std::collections::HashMap;

const DEFAULT_FALLBACK_TOKENS_PER_SECOND: usize = 1000;
const BASELINE_CAPACITY_MILLI: usize = 1000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrefillTimeEstimate {
    pub raw_micros: usize,
    pub effective_micros: usize,
    pub beta_milli: usize,
    pub source: &'static str,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PrefillScoreProfiles {
    profiles: HashMap<String, PrefillCurve>,
    fallback_tokens_per_second: usize,
}

#[derive(Debug, Clone)]
struct PrefillCurve {
    points: Vec<CurvePoint>,
}

#[derive(Debug, Clone)]
struct CurvePoint {
    tokens: usize,
    raw_micros: usize,
    beta_milli: usize,
}

#[derive(Debug, Deserialize)]
struct RawProfiles {
    #[serde(default = "default_fallback_tokens_per_second")]
    fallback_tokens_per_second: usize,
    #[serde(default)]
    profiles: HashMap<String, RawProfile>,
}

#[derive(Debug, Deserialize)]
struct RawProfile {
    /// `[uncached_tokens, prefill_ms]`, strictly increasing in both axes.
    curve_ms: Vec<[f64; 2]>,
    /// Optional `[uncached_tokens, beta]` points. Beta is interpolated by
    /// length and applied only to PD Prefill candidates.
    #[serde(default)]
    pd_beta: Vec<[f64; 2]>,
}

fn default_fallback_tokens_per_second() -> usize {
    DEFAULT_FALLBACK_TOKENS_PER_SECOND
}

impl Default for RawProfiles {
    fn default() -> Self {
        Self {
            fallback_tokens_per_second: default_fallback_tokens_per_second(),
            profiles: HashMap::new(),
        }
    }
}

impl PrefillScoreProfiles {
    pub(crate) fn from_env() -> Self {
        let Some(raw) = std::env::var("PREFILL_SCORE_PROFILES_JSON")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return Self::fallback(DEFAULT_FALLBACK_TOKENS_PER_SECOND);
        };
        match Self::from_json(&raw) {
            Ok(profiles) => profiles,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "invalid PREFILL_SCORE_PROFILES_JSON; using fixed-throughput PrefillScore fallback"
                );
                Self::fallback(DEFAULT_FALLBACK_TOKENS_PER_SECOND)
            }
        }
    }

    pub(crate) fn from_json(raw: &str) -> Result<Self, String> {
        let raw: RawProfiles = serde_json::from_str(raw)
            .map_err(|error| format!("parse PrefillScore profiles JSON: {error}"))?;
        if raw.fallback_tokens_per_second == 0 {
            return Err("fallback_tokens_per_second must be greater than zero".into());
        }
        let profiles = raw
            .profiles
            .into_iter()
            .map(|(selector, profile)| {
                if selector.trim().is_empty() {
                    return Err("profile selector must be non-empty".to_string());
                }
                PrefillCurve::try_from(profile).map(|curve| (selector, curve))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        Ok(Self {
            profiles,
            fallback_tokens_per_second: raw.fallback_tokens_per_second,
        })
    }

    pub(crate) fn fallback(tokens_per_second: usize) -> Self {
        Self {
            profiles: HashMap::new(),
            fallback_tokens_per_second: tokens_per_second.max(1),
        }
    }

    pub(crate) fn estimate(
        &self,
        selector: &str,
        tokens: usize,
        prefill_capacity_milli: usize,
        is_pd_prefill: bool,
    ) -> PrefillTimeEstimate {
        let curve = self
            .profiles
            .get(selector)
            .or_else(|| self.profiles.get("default"));
        if let Some(curve) = curve {
            return curve.estimate(tokens, is_pd_prefill);
        }
        let raw_micros = self.fixed_micros(tokens, prefill_capacity_milli);
        PrefillTimeEstimate {
            raw_micros,
            effective_micros: raw_micros,
            beta_milli: BASELINE_CAPACITY_MILLI,
            source: "fixed-throughput",
        }
    }

    /// Aggregate token summaries have lost the individual length distribution,
    /// so they must stay on the linear fallback instead of being treated as one
    /// large nonlinear request.
    pub(crate) fn fixed_micros(&self, tokens: usize, prefill_capacity_milli: usize) -> usize {
        tokens
            .saturating_mul(1_000_000)
            .saturating_div(self.fallback_tokens_per_second.max(1))
            .saturating_mul(BASELINE_CAPACITY_MILLI)
            .saturating_div(prefill_capacity_milli.max(1))
    }
}

impl TryFrom<RawProfile> for PrefillCurve {
    type Error = String;

    fn try_from(raw: RawProfile) -> Result<Self, Self::Error> {
        if raw.curve_ms.is_empty() {
            return Err("curve_ms must contain at least one point".into());
        }
        validate_points(&raw.curve_ms, "curve_ms", true)?;
        if !raw.pd_beta.is_empty() {
            validate_points(&raw.pd_beta, "pd_beta", false)?;
            if raw.pd_beta.iter().any(|point| point[1] < 1.0) {
                return Err("pd_beta values must be >= 1.0".into());
            }
        }
        let points = raw
            .curve_ms
            .iter()
            .map(|point| {
                let tokens = point[0] as usize;
                let raw_micros = (point[1] * 1000.0).round() as usize;
                let beta = interpolate_f64(&raw.pd_beta, tokens, 1.0);
                CurvePoint {
                    tokens,
                    raw_micros,
                    beta_milli: (beta * 1000.0).round().max(1000.0) as usize,
                }
            })
            .collect::<Vec<_>>();
        if points
            .windows(2)
            .any(|pair| effective_micros(&pair[0]) > effective_micros(&pair[1]))
        {
            return Err("curve_ms / pd_beta produce a non-monotonic effective PD curve".into());
        }
        Ok(Self { points })
    }
}

impl PrefillCurve {
    fn estimate(&self, tokens: usize, is_pd_prefill: bool) -> PrefillTimeEstimate {
        if tokens == 0 {
            return PrefillTimeEstimate {
                raw_micros: 0,
                effective_micros: 0,
                beta_milli: BASELINE_CAPACITY_MILLI,
                source: "profile-curve",
            };
        }
        let raw_micros = interpolate_us(&self.points, tokens, |point| point.raw_micros);
        let beta_milli = if is_pd_prefill {
            interpolate_us(&self.points, tokens, |point| point.beta_milli)
                .max(BASELINE_CAPACITY_MILLI)
        } else {
            BASELINE_CAPACITY_MILLI
        };
        let effective_micros = raw_micros
            .saturating_mul(BASELINE_CAPACITY_MILLI)
            .saturating_div(beta_milli.max(1));
        PrefillTimeEstimate {
            raw_micros,
            effective_micros,
            beta_milli,
            source: "profile-curve",
        }
    }
}

fn validate_points(
    points: &[[f64; 2]],
    name: &str,
    require_increasing_y: bool,
) -> Result<(), String> {
    if points.iter().any(|point| {
        !point[0].is_finite() || !point[1].is_finite() || point[0] < 0.0 || point[1] <= 0.0
    }) {
        return Err(format!(
            "{name} points must contain finite non-negative tokens and positive values"
        ));
    }
    if points
        .windows(2)
        .any(|pair| pair[0][0] >= pair[1][0] || (require_increasing_y && pair[0][1] >= pair[1][1]))
    {
        return Err(format!(
            "{name} token coordinates must be strictly increasing{}",
            if require_increasing_y {
                " and values must be strictly increasing"
            } else {
                ""
            }
        ));
    }
    Ok(())
}

fn interpolate_f64(points: &[[f64; 2]], tokens: usize, default: f64) -> f64 {
    if points.is_empty() {
        return default;
    }
    interpolate(points, tokens as f64, |point| point[0], |point| point[1])
}

fn interpolate_us(
    points: &[CurvePoint],
    tokens: usize,
    value: impl Fn(&CurvePoint) -> usize,
) -> usize {
    interpolate(
        points,
        tokens as f64,
        |point| point.tokens as f64,
        |point| value(point) as f64,
    )
    .round() as usize
}

fn interpolate<T>(
    points: &[T],
    x: f64,
    x_value: impl Fn(&T) -> f64,
    y_value: impl Fn(&T) -> f64,
) -> f64 {
    let first = &points[0];
    if x <= x_value(first) {
        let first_x = x_value(first);
        return if first_x == 0.0 {
            y_value(first)
        } else {
            y_value(first) * x / first_x
        };
    }
    for pair in points.windows(2) {
        let left_x = x_value(&pair[0]);
        let right_x = x_value(&pair[1]);
        if x <= right_x {
            let ratio = (x - left_x) / (right_x - left_x);
            return y_value(&pair[0]) + ratio * (y_value(&pair[1]) - y_value(&pair[0]));
        }
    }
    let last = &points[points.len() - 1];
    if points.len() == 1 {
        return y_value(last) * x / x_value(last).max(1.0);
    }
    let previous = &points[points.len() - 2];
    let slope = (y_value(last) - y_value(previous)) / (x_value(last) - x_value(previous));
    y_value(last) + slope * (x - x_value(last))
}

fn effective_micros(point: &CurvePoint) -> usize {
    point
        .raw_micros
        .saturating_mul(BASELINE_CAPACITY_MILLI)
        .saturating_div(point.beta_milli.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_nonlinear_curve_and_length_dependent_pd_beta() {
        let profiles = PrefillScoreProfiles::from_json(
            r#"{
              "fallback_tokens_per_second": 2000,
              "profiles": {
                "http://p0": {
                  "curve_ms": [[2000, 20], [8000, 200], [32000, 3200]],
                  "pd_beta": [[2000, 2.0], [8000, 4.0], [32000, 8.0]]
                }
              }
            }"#,
        )
        .unwrap();
        let integrated = profiles.estimate("http://p0", 5000, 1000, false);
        let pd = profiles.estimate("http://p0", 5000, 1000, true);
        assert_eq!(integrated.raw_micros, 110_000);
        assert_eq!(integrated.effective_micros, 110_000);
        assert_eq!(pd.beta_milli, 3000);
        assert_eq!(pd.effective_micros, 36_666);
    }

    #[test]
    fn aggregate_fallback_stays_linear_and_capacity_weighted() {
        let profiles = PrefillScoreProfiles::fallback(2000);
        assert_eq!(profiles.fixed_micros(4000, 1000), 2_000_000);
        assert_eq!(profiles.fixed_micros(4000, 2000), 1_000_000);
    }

    #[test]
    fn rejects_effective_curve_that_goes_backwards() {
        let error = PrefillScoreProfiles::from_json(
            r#"{"profiles":{"default":{"curve_ms":[[1000,10],[2000,20]],"pd_beta":[[1000,1],[2000,4]]}}}"#,
        )
        .unwrap_err();
        assert!(error.contains("non-monotonic"));
    }
}
