use crate::models::gguf::{GgufCompactMeta, GgufKvCacheQuant};

const DEFAULT_CONTEXT_LENGTH: u32 = 4096;
const DEFAULT_PARALLEL_SLOTS: usize = 4;
const MIN_AUTO_CONTEXT_LENGTH: u32 = 512;
const MAX_AUTO_PARALLEL_SLOTS: usize = 16;
const KV_CACHE_BUDGET_NUMERATOR: u64 = 85;
const KV_CACHE_BUDGET_DENOMINATOR: u64 = 100;

/// KV cache quantisation levels the planner tries when negotiating context.
/// Ordered from least aggressive to most aggressive compression.
const KV_QUANT_LADDER: &[GgufKvCacheQuant] = &[
    GgufKvCacheQuant::F16,
    GgufKvCacheQuant::Q8_0,
    GgufKvCacheQuant::Q4_0,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeResourcePlan {
    pub(super) context_length: u32,
    pub(super) slots: usize,
    /// The KV cache quantisation the planner selected.  When the planner
    /// negotiates down from f16 to hit the target context length this may
    /// differ from the input `kv_cache_quant`.
    pub(super) negotiated_kv_quant: Option<GgufKvCacheQuant>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RuntimeResourcePlanInput<'a> {
    pub(super) ctx_size_override: Option<u32>,
    pub(super) parallel_override: Option<usize>,
    /// Model weight bytes **local to this node**.  For a whole-model load this
    /// is the full GGUF file size.  For a split/layer-package load, callers
    /// should pass only this node's share of the model weights (i.e.
    /// `source_model_bytes * local_layers / total_layers`).
    pub(super) model_bytes: u64,
    pub(super) vram_bytes: u64,
    pub(super) metadata: Option<&'a GgufCompactMeta>,
    pub(super) kv_cache_quant: GgufKvCacheQuant,
    /// Fraction of the model's layers that reside on this node (0.0–1.0).
    /// When `Some`, the KV cache budget is scaled by this fraction because
    /// each pipeline-parallel stage only holds KV state for its own layers.
    /// `None` means the whole model is local (fraction = 1.0).
    pub(super) local_layer_fraction: Option<f64>,
}

pub(super) fn plan_runtime_resources(input: RuntimeResourcePlanInput<'_>) -> RuntimeResourcePlan {
    let (context_length, negotiated_kv_quant) = if input.ctx_size_override.is_some() {
        (
            input
                .ctx_size_override
                .unwrap_or_else(|| planned_context_length(input, input.kv_cache_quant)),
            None,
        )
    } else {
        negotiate_context_and_kv_quant(input)
    };
    let effective_quant = negotiated_kv_quant.unwrap_or(input.kv_cache_quant);
    let slots = input
        .parallel_override
        .unwrap_or_else(|| planned_parallel_slots(input, effective_quant, context_length));

    RuntimeResourcePlan {
        context_length,
        slots,
        negotiated_kv_quant,
    }
}

/// Try increasingly aggressive KV cache quantisation to maximise context.
///
/// Starts with the caller's requested quant, then walks the quant ladder
/// toward q4_0.  At each step it computes the best affordable context.  The
/// first quant that reaches the model's native context wins.  If none reach
/// native, the quant that produced the largest context is used.
fn negotiate_context_and_kv_quant(
    input: RuntimeResourcePlanInput<'_>,
) -> (u32, Option<GgufKvCacheQuant>) {
    let native_context = input.metadata.map(|m| m.context_length).filter(|c| *c > 0);

    let mut best_quant: Option<GgufKvCacheQuant> = None;

    // Always evaluate the caller-supplied quant first.
    let starting_context = planned_context_length(input, input.kv_cache_quant);
    if native_context.map_or(true, |nc| starting_context >= nc) {
        // Already at native — no negotiation needed.
        return (starting_context, None);
    }
    let mut best_context = starting_context;

    // Walk the ladder from the caller's quant onward (skip levels that are
    // less aggressive than the caller already requested).
    for &quant in KV_QUANT_LADDER {
        if !quant.is_more_aggressive_than(input.kv_cache_quant) {
            continue;
        }
        let ctx = planned_context_length(input, quant);
        if ctx > best_context {
            best_context = ctx;
            best_quant = Some(quant);
        }
        if native_context.map_or(false, |nc| ctx >= nc) {
            // Reached native context — no need to compress further.
            return (ctx, Some(quant));
        }
    }

    if best_quant.is_some() {
        (best_context, best_quant)
    } else {
        (best_context, None)
    }
}

fn planned_context_length(input: RuntimeResourcePlanInput<'_>, kv_quant: GgufKvCacheQuant) -> u32 {
    let Some(metadata) = input.metadata else {
        return DEFAULT_CONTEXT_LENGTH;
    };
    let native_context = metadata.context_length;
    if native_context == 0 {
        return DEFAULT_CONTEXT_LENGTH;
    }
    let Some(kv_bytes_per_token_full) = kv_quant.kv_cache_bytes_per_token(metadata) else {
        return DEFAULT_CONTEXT_LENGTH.min(native_context);
    };

    // In a pipeline-parallel split each stage only holds KV state for its
    // own layers.  Scale the per-token cost by the local layer fraction.
    let layer_fraction = input.local_layer_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    let kv_bytes_per_token = if layer_fraction < 1.0 && layer_fraction > 0.0 {
        ((kv_bytes_per_token_full as f64) * layer_fraction).ceil() as u64
    } else {
        kv_bytes_per_token_full
    };

    let kv_budget = usable_kv_cache_budget(input.vram_bytes, input.model_bytes);
    if kv_bytes_per_token == 0 {
        return native_context;
    }
    let max_affordable_context = kv_budget / kv_bytes_per_token;
    if max_affordable_context == 0 {
        return MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    }

    let planned = max_affordable_context
        .min(u64::from(native_context))
        .min(u64::from(u32::MAX)) as u32;
    let minimum = MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    if planned < minimum {
        minimum
    } else {
        snap_context_length_down(planned).max(minimum)
    }
}

fn planned_parallel_slots(
    input: RuntimeResourcePlanInput<'_>,
    kv_quant: GgufKvCacheQuant,
    context_length: u32,
) -> usize {
    let Some(metadata) = input.metadata else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    let Some(kv_bytes_per_token_full) = kv_quant.kv_cache_bytes_per_token(metadata) else {
        return DEFAULT_PARALLEL_SLOTS;
    };

    let layer_fraction = input.local_layer_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    let kv_bytes_per_token = if layer_fraction < 1.0 && layer_fraction > 0.0 {
        ((kv_bytes_per_token_full as f64) * layer_fraction).ceil() as u64
    } else {
        kv_bytes_per_token_full
    };

    let Some(bytes_per_slot) = u64::from(context_length).checked_mul(kv_bytes_per_token) else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    if bytes_per_slot == 0 {
        return DEFAULT_PARALLEL_SLOTS;
    }

    let raw_slots = usable_kv_cache_budget(input.vram_bytes, input.model_bytes) / bytes_per_slot;
    snap_parallel_slots_down(raw_slots)
}

fn usable_kv_cache_budget(vram_bytes: u64, model_bytes: u64) -> u64 {
    let free_bytes = vram_bytes.saturating_sub(model_bytes);
    let budget = u128::from(free_bytes) * u128::from(KV_CACHE_BUDGET_NUMERATOR)
        / u128::from(KV_CACHE_BUDGET_DENOMINATOR);
    budget.min(u128::from(u64::MAX)) as u64
}

fn snap_parallel_slots_down(raw_slots: u64) -> usize {
    match raw_slots.min(MAX_AUTO_PARALLEL_SLOTS as u64) {
        0 => 1,
        1 => 1,
        2 | 3 => 2,
        4..=7 => 4,
        8..=15 => 8,
        _ => MAX_AUTO_PARALLEL_SLOTS,
    }
}

fn snap_context_length_down(value: u32) -> u32 {
    const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];
    CONTEXT_STEPS
        .iter()
        .rev()
        .copied()
        .find(|step| *step <= value)
        .unwrap_or(MIN_AUTO_CONTEXT_LENGTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gqa_metadata(context_length: u32) -> GgufCompactMeta {
        GgufCompactMeta {
            context_length,
            head_count: 32,
            kv_head_count: 8,
            layer_count: 32,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        }
    }

    #[test]
    fn plan_runtime_resources_preserves_explicit_overrides() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: Some(16_384),
            parallel_override: Some(7),
            model_bytes: 10_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 16_384);
        assert_eq!(plan.slots, 7);
        assert_eq!(plan.negotiated_kv_quant, None);
    }

    #[test]
    fn plan_runtime_resources_clamps_auto_context_to_native_metadata() {
        let metadata = gqa_metadata(16_384);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 16_384);
        assert!(
            plan.slots > 4,
            "expected metadata-based slots, got {plan:?}"
        );
    }

    #[test]
    fn plan_runtime_resources_snaps_auto_context_down() {
        let metadata = gqa_metadata(32_768);
        // With KV negotiation enabled, the planner starts at f16 (8K), sees it
        // can't reach 32K native, and tries q4_0 which affords 16K.
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(1),
            model_bytes: 5_000_000_000,
            vram_bytes: 6_300_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 16_384);
        assert_eq!(plan.slots, 1);
        assert!(
            plan.negotiated_kv_quant.is_some(),
            "planner should negotiate KV quant to reach larger context"
        );
    }

    #[test]
    fn plan_runtime_resources_uses_effective_kv_quant_for_slot_budget() {
        let metadata = gqa_metadata(131_072);
        let f16_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });
        let q4_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q4_0,
            local_layer_fraction: None,
        });

        assert_eq!(f16_plan.context_length, q4_plan.context_length);
        assert!(
            q4_plan.slots > f16_plan.slots,
            "expected quantized KV cache to allow more slots: f16={f16_plan:?}, q4={q4_plan:?}"
        );
    }

    #[test]
    fn plan_runtime_resources_falls_back_to_legacy_defaults_without_metadata() {
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: None,
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        assert_eq!(
            plan,
            RuntimeResourcePlan {
                context_length: 4096,
                slots: 4,
                negotiated_kv_quant: None,
            }
        );
    }

    #[test]
    fn plan_runtime_resources_uses_explicit_parallel_with_metadata_context() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(2),
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 32_768);
        assert_eq!(plan.slots, 2);
    }

    // --- New tests for split-aware planning and KV negotiation ---

    #[test]
    fn split_model_uses_local_layer_fraction_for_budget() {
        // 480B-class model: 94 layers, 264GB total, node holds 62/94 layers
        let metadata = GgufCompactMeta {
            context_length: 131_072,
            head_count: 64,
            kv_head_count: 8,
            layer_count: 94,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let total_model_bytes: u64 = 264_000_000_000;
        let local_fraction = 62.0 / 94.0;
        let local_model_bytes = (total_model_bytes as f64 * local_fraction) as u64;

        // Without split awareness: local VRAM 206 GB, total model 264 GB → negative budget
        let no_split_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: total_model_bytes,
            vram_bytes: 206_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q4_0,
            local_layer_fraction: None,
        });

        // With split awareness: local model ~174 GB, local KV fraction 0.66
        let split_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: local_model_bytes,
            vram_bytes: 206_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q4_0,
            local_layer_fraction: Some(local_fraction),
        });

        assert!(
            split_plan.context_length > no_split_plan.context_length,
            "split-aware planning should produce larger context: split={}, no_split={}",
            split_plan.context_length,
            no_split_plan.context_length
        );
        assert!(
            split_plan.context_length >= 65_536,
            "480B split across 206+103 GB should achieve at least 64K context with q4_0, got {}",
            split_plan.context_length
        );
    }

    #[test]
    fn negotiate_kv_quant_upgrades_to_reach_native_context() {
        // Model with 32K native context.  With f16 KV only 8K fits, but q4_0
        // should reach 32K.
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 6_300_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        // With negotiation the planner should try q8_0 / q4_0 and reach 32K.
        assert!(
            plan.context_length > 8192,
            "negotiation should improve context beyond f16-only: got {}",
            plan.context_length
        );
        assert!(
            plan.negotiated_kv_quant.is_some(),
            "planner should have negotiated a more aggressive KV quant"
        );
    }

    #[test]
    fn negotiate_kv_quant_does_not_downgrade_when_already_at_native() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 32_768);
        assert_eq!(
            plan.negotiated_kv_quant, None,
            "should not negotiate when f16 already reaches native context"
        );
    }
}
