use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};

use super::SpdHeadTopology;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpdStageLayerRange {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
}

impl SpdStageLayerRange {
    pub const fn new(stage_index: u32, layer_start: u32, layer_end: u32) -> Self {
        Self {
            stage_index,
            layer_start,
            layer_end,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpdHiddenStateSource {
    Embedding,
    StageBoundary { stage_index: u32, layer_end: u32 },
    InternalTap { stage_index: u32, layer_after: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpdHiddenStateRequirement {
    pub hf_index: u32,
    pub source: SpdHiddenStateSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdHiddenTapPlan {
    pub required_hf_indices: Vec<u32>,
    pub stage_boundary_hf_indices: Vec<u32>,
    pub requirements: Vec<SpdHiddenStateRequirement>,
    pub boundary_only_missing_hf_indices: Vec<u32>,
}

impl SpdHiddenTapPlan {
    pub fn can_use_stage_boundaries_only(&self) -> bool {
        self.boundary_only_missing_hf_indices.is_empty()
    }

    pub fn requires_internal_taps(&self) -> bool {
        !self.can_use_stage_boundaries_only()
    }
}

pub fn plan_hidden_state_taps(
    topology: &SpdHeadTopology,
    stage_ranges: &[SpdStageLayerRange],
) -> Result<SpdHiddenTapPlan> {
    let stage_ranges = validate_stage_ranges(stage_ranges)?;
    let required_hf_indices = required_hf_indices(topology);
    let stage_boundary_hf_indices = stage_boundary_hf_indices(&stage_ranges);

    let mut requirements = Vec::with_capacity(required_hf_indices.len());
    let mut boundary_only_missing_hf_indices = Vec::new();
    for hf_index in &required_hf_indices {
        let source = source_for_hf_index(*hf_index, &stage_ranges)?;
        if matches!(source, SpdHiddenStateSource::InternalTap { .. }) {
            boundary_only_missing_hf_indices.push(*hf_index);
        }
        requirements.push(SpdHiddenStateRequirement {
            hf_index: *hf_index,
            source,
        });
    }

    Ok(SpdHiddenTapPlan {
        required_hf_indices,
        stage_boundary_hf_indices,
        requirements,
        boundary_only_missing_hf_indices,
    })
}

fn validate_stage_ranges(stage_ranges: &[SpdStageLayerRange]) -> Result<Vec<SpdStageLayerRange>> {
    if stage_ranges.is_empty() {
        bail!("SPD hidden-state tap planning requires at least one stage range");
    }
    let mut seen_stage_indices = BTreeSet::new();
    for range in stage_ranges {
        if range.layer_end <= range.layer_start {
            bail!(
                "SPD stage {} has empty layer range {}..{}",
                range.stage_index,
                range.layer_start,
                range.layer_end
            );
        }
        if !seen_stage_indices.insert(range.stage_index) {
            bail!("duplicate SPD stage index {}", range.stage_index);
        }
    }

    let mut sorted = stage_ranges.to_vec();
    sorted.sort_by_key(|range| (range.layer_start, range.layer_end, range.stage_index));
    for pair in sorted.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if left.layer_end > right.layer_start {
            bail!(
                "SPD stage ranges overlap: stage {} is {}..{}, stage {} is {}..{}",
                left.stage_index,
                left.layer_start,
                left.layer_end,
                right.stage_index,
                right.layer_start,
                right.layer_end
            );
        }
    }
    Ok(sorted)
}

fn required_hf_indices(topology: &SpdHeadTopology) -> Vec<u32> {
    let mut indices = BTreeSet::new();
    for group in &topology.shallow_hidden_layer_indices {
        indices.extend(group.iter().copied());
    }
    indices.into_iter().collect()
}

fn stage_boundary_hf_indices(stage_ranges: &[SpdStageLayerRange]) -> Vec<u32> {
    stage_ranges
        .iter()
        .map(|range| range.layer_end)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn source_for_hf_index(
    hf_index: u32,
    stage_ranges: &[SpdStageLayerRange],
) -> Result<SpdHiddenStateSource> {
    if hf_index == 0 {
        return Ok(SpdHiddenStateSource::Embedding);
    }
    if let Some(range) = stage_ranges
        .iter()
        .find(|range| range.layer_end == hf_index)
        .copied()
    {
        return Ok(SpdHiddenStateSource::StageBoundary {
            stage_index: range.stage_index,
            layer_end: range.layer_end,
        });
    }

    let layer_after = hf_index
        .checked_sub(1)
        .context("SPD HF hidden-state index underflow")?;
    if let Some(range) = stage_ranges
        .iter()
        .find(|range| range.layer_start <= layer_after && layer_after < range.layer_end)
        .copied()
    {
        return Ok(SpdHiddenStateSource::InternalTap {
            stage_index: range.stage_index,
            layer_after,
        });
    }

    bail!(
        "SPD HF hidden-state index {hf_index} has no owning stage range; index k>=1 means output after layer k-1"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen35_s4_l4_topology() -> SpdHeadTopology {
        SpdHeadTopology {
            hidden_size: 2560,
            vocab_size: 248_320,
            draft_vocab_size: 50_000,
            num_stages: 4,
            stage_layer_boundaries: Some(vec![8, 16, 24, 32]),
            num_spec_layers: 4,
            trained_with_use_deepest: true,
            shallow_hidden_layer_indices: vec![
                vec![0, 10, 20, 31],
                vec![0, 8, 16, 24],
                vec![0, 8, 16],
                vec![0, 8],
            ],
            spec_init_from_base_layers: None,
            draft_token_ids: None,
            rope_theta: None,
            rotary_dim: None,
        }
    }

    fn qwen35_s2_l4_topology() -> SpdHeadTopology {
        SpdHeadTopology {
            hidden_size: 2560,
            vocab_size: 248_320,
            draft_vocab_size: 50_000,
            num_stages: 2,
            stage_layer_boundaries: Some(vec![16, 32]),
            num_spec_layers: 4,
            trained_with_use_deepest: true,
            shallow_hidden_layer_indices: vec![vec![0, 16, 32], vec![0, 16]],
            spec_init_from_base_layers: None,
            draft_token_ids: None,
            rope_theta: None,
            rotary_dim: None,
        }
    }

    #[test]
    fn qwen35_pretrained_head_needs_internal_taps_on_even_four_way_split() {
        let plan = plan_hidden_state_taps(
            &qwen35_s4_l4_topology(),
            &[
                SpdStageLayerRange::new(0, 0, 8),
                SpdStageLayerRange::new(1, 8, 16),
                SpdStageLayerRange::new(2, 16, 24),
                SpdStageLayerRange::new(3, 24, 32),
            ],
        )
        .unwrap();

        assert_eq!(plan.required_hf_indices, vec![0, 8, 10, 16, 20, 24, 31]);
        assert_eq!(plan.stage_boundary_hf_indices, vec![8, 16, 24, 32]);
        assert_eq!(plan.boundary_only_missing_hf_indices, vec![10, 20, 31]);
        assert!(plan.requires_internal_taps());
        assert_eq!(
            plan.requirements
                .iter()
                .find(|requirement| requirement.hf_index == 10)
                .unwrap()
                .source,
            SpdHiddenStateSource::InternalTap {
                stage_index: 1,
                layer_after: 9
            }
        );
    }

    #[test]
    fn qwen35_pretrained_head_can_be_boundary_only_with_tap_aligned_split() {
        let plan = plan_hidden_state_taps(
            &qwen35_s4_l4_topology(),
            &[
                SpdStageLayerRange::new(0, 0, 8),
                SpdStageLayerRange::new(1, 8, 10),
                SpdStageLayerRange::new(2, 10, 16),
                SpdStageLayerRange::new(3, 16, 20),
                SpdStageLayerRange::new(4, 20, 24),
                SpdStageLayerRange::new(5, 24, 31),
                SpdStageLayerRange::new(6, 31, 32),
            ],
        )
        .unwrap();

        assert!(plan.can_use_stage_boundaries_only());
        assert_eq!(plan.boundary_only_missing_hf_indices, Vec::<u32>::new());
        assert_eq!(
            plan.requirements
                .iter()
                .find(|requirement| requirement.hf_index == 31)
                .unwrap()
                .source,
            SpdHiddenStateSource::StageBoundary {
                stage_index: 5,
                layer_end: 31
            }
        );
    }

    #[test]
    fn qwen35_two_stage_head_can_use_boundary_only_split() {
        let plan = plan_hidden_state_taps(
            &qwen35_s2_l4_topology(),
            &[
                SpdStageLayerRange::new(0, 0, 16),
                SpdStageLayerRange::new(1, 16, 32),
            ],
        )
        .unwrap();

        assert!(plan.can_use_stage_boundaries_only());
        assert_eq!(plan.required_hf_indices, vec![0, 16, 32]);
        assert_eq!(plan.stage_boundary_hf_indices, vec![16, 32]);
        assert_eq!(plan.boundary_only_missing_hf_indices, Vec::<u32>::new());
        assert_eq!(
            plan.requirements
                .iter()
                .find(|requirement| requirement.hf_index == 0)
                .unwrap()
                .source,
            SpdHiddenStateSource::Embedding
        );
        assert_eq!(
            plan.requirements
                .iter()
                .find(|requirement| requirement.hf_index == 16)
                .unwrap()
                .source,
            SpdHiddenStateSource::StageBoundary {
                stage_index: 0,
                layer_end: 16
            }
        );
        assert_eq!(
            plan.requirements
                .iter()
                .find(|requirement| requirement.hf_index == 32)
                .unwrap()
                .source,
            SpdHiddenStateSource::StageBoundary {
                stage_index: 1,
                layer_end: 32
            }
        );
    }

    #[test]
    fn rejects_required_tap_without_owner_stage() {
        let error = plan_hidden_state_taps(
            &qwen35_s4_l4_topology(),
            &[
                SpdStageLayerRange::new(0, 0, 8),
                SpdStageLayerRange::new(1, 8, 16),
            ],
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("HF hidden-state index 20"));
        assert!(error.contains("output after layer k-1"));
    }
}
