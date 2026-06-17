use super::NativeMtpAdaptiveDisableConfig;

const BATCHED_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_BATCHED_VERIFY";
const ADAPTIVE_DISABLE_ENV: &str = "SKIPPY_NATIVE_MTP_ADAPTIVE_DISABLE";
const ADAPTIVE_DISABLE_MIN_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_ADAPTIVE_DISABLE_MIN_VERIFY";
const ADAPTIVE_DISABLE_THRESHOLD_ENV: &str = "SKIPPY_NATIVE_MTP_ADAPTIVE_DISABLE_THRESHOLD";
const REJECT_COOLDOWN_TOKENS_ENV: &str = "SKIPPY_NATIVE_MTP_REJECT_COOLDOWN_TOKENS";
const REJECT_RECOVERY_SERIAL_ACCEPTS_ENV: &str = "SKIPPY_NATIVE_MTP_REJECT_RECOVERY_SERIAL_ACCEPTS";
const SERIAL_AFTER_GAP_REJECT_RECOVERY_SERIAL_ACCEPTS_ENV: &str =
    "SKIPPY_NATIVE_MTP_SERIAL_AFTER_GAP_REJECT_RECOVERY_SERIAL_ACCEPTS";
const VERIFY_NEXT_REJECT_RECOVERY_SERIAL_ACCEPTS_ENV: &str =
    "SKIPPY_NATIVE_MTP_VERIFY_NEXT_REJECT_RECOVERY_SERIAL_ACCEPTS";
const SERIAL_AFTER_GAP_DRAFT_MIN_MARGIN_ENV: &str =
    "SKIPPY_NATIVE_MTP_SERIAL_AFTER_GAP_DRAFT_MIN_MARGIN";
const VERIFY_NEXT_DRAFT_MIN_MARGIN_ENV: &str = "SKIPPY_NATIVE_MTP_VERIFY_NEXT_DRAFT_MIN_MARGIN";
const DEFER_REJECT_TRIM_ENV: &str = "SKIPPY_NATIVE_MTP_DEFER_REJECT_TRIM";
const SUPPRESS_COOLDOWN_DRAFTS_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFTS";
const SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFT_LIMIT";
const DEFAULT_ADAPTIVE_DISABLE_MIN_VERIFY: u64 = 32;
const DEFAULT_ADAPTIVE_DISABLE_THRESHOLD: f64 = 0.70;

pub(in crate::frontend) fn native_mtp_batched_verify_enabled() -> bool {
    native_mtp_batched_verify_enabled_from(std::env::var(BATCHED_VERIFY_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_adaptive_disable_config() -> NativeMtpAdaptiveDisableConfig {
    NativeMtpAdaptiveDisableConfig {
        enabled: truthy_env(std::env::var(ADAPTIVE_DISABLE_ENV).ok().as_deref()),
        min_verifications: parse_u64_env(
            ADAPTIVE_DISABLE_MIN_VERIFY_ENV,
            DEFAULT_ADAPTIVE_DISABLE_MIN_VERIFY,
        ),
        threshold: parse_threshold_env(
            ADAPTIVE_DISABLE_THRESHOLD_ENV,
            DEFAULT_ADAPTIVE_DISABLE_THRESHOLD,
        ),
    }
}

pub(in crate::frontend) fn native_mtp_reject_cooldown_tokens() -> usize {
    parse_usize_env(REJECT_COOLDOWN_TOKENS_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_reject_recovery_serial_accepts() -> usize {
    parse_usize_env(REJECT_RECOVERY_SERIAL_ACCEPTS_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_serial_after_gap_reject_recovery_serial_accepts() -> usize {
    parse_usize_env(SERIAL_AFTER_GAP_REJECT_RECOVERY_SERIAL_ACCEPTS_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_verify_next_reject_recovery_serial_accepts() -> usize {
    parse_usize_env(VERIFY_NEXT_REJECT_RECOVERY_SERIAL_ACCEPTS_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_serial_after_gap_draft_min_margin() -> Option<f32> {
    parse_optional_f32_env(SERIAL_AFTER_GAP_DRAFT_MIN_MARGIN_ENV)
}

pub(in crate::frontend) fn native_mtp_verify_next_draft_min_margin() -> Option<f32> {
    parse_optional_f32_env(VERIFY_NEXT_DRAFT_MIN_MARGIN_ENV)
}

pub(in crate::frontend) fn native_mtp_defer_reject_trim_enabled() -> bool {
    truthy_env(std::env::var(DEFER_REJECT_TRIM_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_suppress_cooldown_drafts_enabled() -> bool {
    truthy_env(std::env::var(SUPPRESS_COOLDOWN_DRAFTS_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_suppress_cooldown_draft_limit() -> usize {
    parse_usize_env(SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV, 0)
}

fn native_mtp_batched_verify_enabled_from(value: Option<&str>) -> bool {
    !falsey_env(value)
}

fn truthy_env(value: Option<&str>) -> bool {
    matches!(
        normalized_env(value).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn falsey_env(value: Option<&str>) -> bool {
    matches!(
        normalized_env(value).as_deref(),
        Some("0" | "false" | "off" | "disable" | "disabled" | "no")
    )
}

fn normalized_env(value: Option<&str>) -> Option<String> {
    value.map(str::trim).map(str::to_ascii_lowercase)
}

fn parse_u64_env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_threshold_env(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| (0.0..=1.0).contains(value))
        .unwrap_or(default)
}

fn parse_optional_f32_env(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_verify_flag_defaults_on_and_accepts_false_values() {
        assert!(native_mtp_batched_verify_enabled_from(None));
        assert!(native_mtp_batched_verify_enabled_from(Some("1")));
        assert!(native_mtp_batched_verify_enabled_from(Some("true")));
        assert!(!native_mtp_batched_verify_enabled_from(Some("0")));
        assert!(!native_mtp_batched_verify_enabled_from(Some("false")));
        assert!(!native_mtp_batched_verify_enabled_from(Some(" disabled ")));
    }

    #[test]
    fn truthy_env_accepts_enabled_aliases_only() {
        for value in ["1", "true", " enabled ", "yes", "on"] {
            assert!(truthy_env(Some(value)), "{value}");
        }
        for value in [
            None,
            Some("0"),
            Some("false"),
            Some("off"),
            Some("disabled"),
        ] {
            assert!(!truthy_env(value), "{value:?}");
        }
    }

    #[test]
    fn parse_threshold_rejects_invalid_ranges() {
        assert_eq!(
            parse_threshold_env("SKIPPY_TEST_MISSING_THRESHOLD", 0.7),
            0.7
        );
    }

    #[test]
    fn numeric_options_default_when_absent() {
        assert_eq!(parse_usize_env("SKIPPY_TEST_MISSING_REJECT_COOLDOWN", 0), 0);
        assert_eq!(parse_usize_env("SKIPPY_TEST_MISSING_REJECT_RECOVERY", 0), 0);
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_GAP_REJECT_RECOVERY", 0),
            0
        );
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_VERIFY_NEXT_REJECT_RECOVERY", 0),
            0
        );
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_SUPPRESS_COOLDOWN_LIMIT", 0),
            0
        );
    }

    #[test]
    fn optional_margins_default_none() {
        assert_eq!(
            parse_optional_f32_env("SKIPPY_TEST_MISSING_GAP_DRAFT_MARGIN"),
            None
        );
        assert_eq!(
            parse_optional_f32_env("SKIPPY_TEST_MISSING_VERIFY_NEXT_MARGIN"),
            None
        );
    }
}
