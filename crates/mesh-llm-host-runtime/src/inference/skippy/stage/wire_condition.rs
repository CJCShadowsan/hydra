use std::env;

use anyhow::{Context, Result, bail};
use skippy_server::binary_transport::WireCondition;

const STAGE_DOWNSTREAM_WIRE_DELAY_MS_ENV: &str = "MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS";
const STAGE_DOWNSTREAM_WIRE_JITTER_MS_ENV: &str = "MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS";
const STAGE_DOWNSTREAM_WIRE_MBPS_ENV: &str = "MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS";

#[derive(Debug)]
pub(super) struct StageDownstreamWireCondition {
    pub(super) condition: WireCondition,
    pub(super) delay_ms: f64,
    pub(super) jitter_ms: f64,
    pub(super) mbps: Option<f64>,
}

impl StageDownstreamWireCondition {
    pub(super) fn enabled(&self) -> bool {
        self.delay_ms > 0.0 || self.jitter_ms > 0.0 || self.mbps.is_some()
    }
}

#[derive(Debug, PartialEq)]
pub(super) struct StageDownstreamWireConfig {
    pub(super) delay_ms: f64,
    pub(super) jitter_ms: f64,
    pub(super) mbps: Option<f64>,
}

pub(super) fn stage_downstream_wire_condition_from_env() -> Result<StageDownstreamWireCondition> {
    let config = parse_stage_downstream_wire_config(
        env::var(STAGE_DOWNSTREAM_WIRE_DELAY_MS_ENV).ok().as_deref(),
        env::var(STAGE_DOWNSTREAM_WIRE_JITTER_MS_ENV)
            .ok()
            .as_deref(),
        env::var(STAGE_DOWNSTREAM_WIRE_MBPS_ENV).ok().as_deref(),
    )?;
    let condition = WireCondition::with_jitter(config.delay_ms, config.jitter_ms, config.mbps)?;
    Ok(StageDownstreamWireCondition {
        condition,
        delay_ms: config.delay_ms,
        jitter_ms: config.jitter_ms,
        mbps: config.mbps,
    })
}

pub(crate) fn stage_downstream_wire_condition_value_from_env() -> Result<WireCondition> {
    Ok(stage_downstream_wire_condition_from_env()?.condition)
}

pub(super) fn parse_stage_downstream_wire_config(
    delay_ms: Option<&str>,
    jitter_ms: Option<&str>,
    mbps: Option<&str>,
) -> Result<StageDownstreamWireConfig> {
    Ok(StageDownstreamWireConfig {
        delay_ms: parse_optional_f64(delay_ms, STAGE_DOWNSTREAM_WIRE_DELAY_MS_ENV, Some(0.0))?
            .unwrap_or(0.0),
        jitter_ms: parse_optional_f64(jitter_ms, STAGE_DOWNSTREAM_WIRE_JITTER_MS_ENV, Some(0.0))?
            .unwrap_or(0.0),
        mbps: parse_optional_f64(mbps, STAGE_DOWNSTREAM_WIRE_MBPS_ENV, None)?,
    })
}

fn parse_optional_f64(
    value: Option<&str>,
    env_name: &str,
    default: Option<f64>,
) -> Result<Option<f64>> {
    let Some(raw) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(default);
    };
    let parsed = raw
        .parse::<f64>()
        .with_context(|| format!("parse {env_name} as f64"))?;
    if !parsed.is_finite() {
        bail!("{env_name} must be finite");
    }
    Ok(Some(parsed))
}
