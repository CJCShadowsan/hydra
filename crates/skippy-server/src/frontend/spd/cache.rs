use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SpdInlineTapRecord {
    pub(super) hf_index: u32,
    pub(super) positions: Vec<u32>,
    pub(super) rows_recorded: usize,
    pub(super) cached_rows: usize,
    pub(super) payload_bytes: usize,
    pub(super) required: bool,
}

pub(super) struct SpdInlineTapCache {
    hidden_size: usize,
    required_hf_indices: BTreeSet<u32>,
    frames: BTreeMap<u32, SpdCachedTapFrame>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SpdTapRecordOutcome {
    Recorded(SpdInlineTapRecord),
    Ignored(SpdIgnoredTap),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SpdIgnoredTap {
    pub(super) reason: &'static str,
    pub(super) positions: Vec<i32>,
    pub(super) accepted_context_len: usize,
    pub(super) pending_positions: Vec<usize>,
}

#[derive(Debug, Default)]
pub(super) struct SpdInlineTapLifecycle {
    pub(super) accepted_context_len: usize,
    pending_future_positions: BTreeSet<usize>,
}

impl SpdInlineTapLifecycle {
    pub(super) fn accept_context_len(&mut self, context_len: usize) {
        self.accepted_context_len = context_len;
        self.pending_future_positions
            .retain(|position| *position >= context_len);
    }

    pub(super) fn reset_context_len(&mut self, context_len: usize) {
        self.accepted_context_len = context_len;
        self.pending_future_positions.clear();
    }

    pub(super) fn mark_pending_optimistic_position(&mut self, position: usize) {
        self.pending_future_positions.insert(position);
    }

    pub(super) fn mark_pending_future_positions(
        &mut self,
        positions: impl IntoIterator<Item = usize>,
    ) {
        self.pending_future_positions.extend(positions);
    }

    pub(super) fn record_decision(&self, tap: &StageReplySpdTap) -> Result<SpdTapRecordDecision> {
        let positions = tap.positions.clone();
        for position in &positions {
            let position = usize::try_from(*position)
                .with_context(|| format!("negative SPD returned tap position {position}"))?;
            if position < self.accepted_context_len
                || self.pending_future_positions.contains(&position)
            {
                continue;
            }
            return Ok(SpdTapRecordDecision {
                ignored: Some(SpdIgnoredTap {
                    reason: "future_position_without_pending_optimistic_context",
                    positions,
                    accepted_context_len: self.accepted_context_len,
                    pending_positions: self.pending_future_positions.iter().copied().collect(),
                }),
            });
        }
        Ok(SpdTapRecordDecision { ignored: None })
    }
}

pub(super) struct SpdTapRecordDecision {
    pub(super) ignored: Option<SpdIgnoredTap>,
}

impl SpdInlineTapCache {
    pub(super) fn new(hidden_size: usize, required_hf_indices: Vec<u32>) -> Self {
        Self {
            hidden_size,
            required_hf_indices: required_hf_indices.into_iter().collect(),
            frames: BTreeMap::new(),
        }
    }

    pub(super) fn retain_positions_before(&mut self, position_limit: usize) {
        self.retain_positions_before_or_in(position_limit, &BTreeSet::new());
    }

    pub(super) fn retain_positions_before_or_in(
        &mut self,
        position_limit: usize,
        retained_positions: &BTreeSet<usize>,
    ) {
        for frame in self.frames.values_mut() {
            frame.rows.retain(|position, _| {
                usize::try_from(*position).is_ok_and(|position| {
                    position < position_limit || retained_positions.contains(&position)
                })
            });
        }
        self.frames.retain(|_, frame| !frame.rows.is_empty());
    }

    pub(super) fn record_stage_output(
        &mut self,
        config: &StageConfig,
        message: &StageWireMessage,
        frame: &ActivationFrame,
    ) -> Result<Option<SpdInlineTapRecord>> {
        let hf_index = config.layer_end;
        if frame.payload.is_empty() {
            return Ok(None);
        }
        validate_spd_inline_frame(frame, self.hidden_size)?;
        let token_count =
            usize::try_from(frame.desc.token_count).context("SPD tap token_count exceeds usize")?;
        if token_count == 0 {
            return Ok(None);
        }
        let positions = message_positions(message, token_count)?;
        self.record_rows(hf_index, positions, frame).map(Some)
    }

    pub(super) fn record_returned_tap(
        &mut self,
        tap: &StageReplySpdTap,
    ) -> Result<SpdInlineTapRecord> {
        if tap.dtype != RuntimeActivationDType::F32 as i32 {
            bail!("SPD returned tap frame must be f32, got {}", tap.dtype);
        }
        if tap.layout != RuntimeActivationLayout::TokenMajor as i32 {
            bail!(
                "SPD returned tap frame must be token-major, got {}",
                tap.layout
            );
        }
        let frame = ActivationFrame {
            desc: skippy_runtime::ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: tap.producer_stage_index,
                layer_start: tap.layer_start,
                layer_end: tap.layer_end,
                token_count: tap.token_count,
                sequence_count: tap.sequence_count,
                payload_bytes: u64::try_from(tap.payload.len())
                    .context("SPD returned tap payload bytes exceed u64")?,
                flags: tap.flags,
            },
            payload: tap.payload.clone(),
        };
        let positions = tap
            .positions
            .iter()
            .copied()
            .map(|position| {
                u32::try_from(position)
                    .with_context(|| format!("negative SPD returned tap position {position}"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.record_rows(tap.hf_index, positions, &frame)
    }

    fn record_rows(
        &mut self,
        hf_index: u32,
        positions: Vec<u32>,
        frame: &ActivationFrame,
    ) -> Result<SpdInlineTapRecord> {
        validate_spd_inline_frame(frame, self.hidden_size)?;
        let token_count =
            usize::try_from(frame.desc.token_count).context("SPD tap token_count exceeds usize")?;
        if positions.len() != token_count {
            bail!(
                "SPD inline tap positions length {} does not match token_count {}",
                positions.len(),
                token_count
            );
        }
        if positions.is_empty() {
            bail!("SPD inline tap has no positions");
        }
        let required = self.required_hf_indices.contains(&hf_index);
        let cached = self
            .frames
            .entry(hf_index)
            .or_insert_with(|| SpdCachedTapFrame::new(frame.desc));
        let row_bytes = self
            .hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .context("SPD inline tap row byte width overflow")?;
        for (row_index, position) in positions.iter().copied().enumerate() {
            let offset = row_index
                .checked_mul(row_bytes)
                .context("SPD inline tap payload offset overflow")?;
            cached
                .rows
                .insert(position, frame.payload[offset..offset + row_bytes].to_vec());
        }
        let rows_recorded = positions.len();
        Ok(SpdInlineTapRecord {
            hf_index,
            positions,
            rows_recorded,
            cached_rows: cached.rows.len(),
            payload_bytes: frame.payload.len(),
            required,
        })
    }

    #[cfg(test)]
    pub(super) fn overlay_complete_frames(
        &mut self,
        taps: &mut BTreeMap<u32, ActivationFrame>,
        row_positions: &[i64],
        required_hf_indices: &[u32],
        hidden_size: usize,
    ) -> Result<()> {
        if hidden_size != self.hidden_size {
            bail!(
                "SPD inline tap hidden size mismatch: cache {}, request {}",
                self.hidden_size,
                hidden_size
            );
        }
        for hf_index in required_hf_indices {
            let Some(frame) = self.frame_for_positions(*hf_index, row_positions)? else {
                continue;
            };
            taps.insert(*hf_index, frame);
        }
        Ok(())
    }

    pub(super) fn overlay_complete_frames_for_row_hf_indices(
        &mut self,
        taps: &mut BTreeMap<u32, ActivationFrame>,
        row_positions: &[i64],
        row_hf_indices: &[Vec<u32>],
        required_hf_indices: &[u32],
        hidden_size: usize,
    ) -> Result<()> {
        if hidden_size != self.hidden_size {
            bail!(
                "SPD inline tap hidden size mismatch: cache {}, request {}",
                self.hidden_size,
                hidden_size
            );
        }
        for (hf_index, positions) in
            required_positions_by_hf_index(row_positions, row_hf_indices, required_hf_indices)?
        {
            let Some(frame) = self.frame_for_positions(hf_index, &positions)? else {
                continue;
            };
            taps.insert(hf_index, frame);
        }
        Ok(())
    }

    pub(super) fn complete_frames(
        &self,
        row_positions: &[i64],
        required_hf_indices: &[u32],
        hidden_size: usize,
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        if hidden_size != self.hidden_size {
            bail!(
                "SPD inline tap hidden size mismatch: cache {}, request {}",
                self.hidden_size,
                hidden_size
            );
        }
        let mut complete = BTreeMap::new();
        for hf_index in required_hf_indices {
            let Some(frame) = self.frame_for_positions(*hf_index, row_positions)? else {
                return Ok(None);
            };
            complete.insert(*hf_index, frame);
        }
        Ok(Some(complete))
    }

    pub(super) fn complete_frames_for_row_hf_indices(
        &self,
        row_positions: &[i64],
        row_hf_indices: &[Vec<u32>],
        required_hf_indices: &[u32],
        hidden_size: usize,
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        if hidden_size != self.hidden_size {
            bail!(
                "SPD inline tap hidden size mismatch: cache {}, request {}",
                self.hidden_size,
                hidden_size
            );
        }
        let mut complete = BTreeMap::new();
        for (hf_index, positions) in
            required_positions_by_hf_index(row_positions, row_hf_indices, required_hf_indices)?
        {
            let Some(frame) = self.frame_for_positions(hf_index, &positions)? else {
                return Ok(None);
            };
            complete.insert(hf_index, frame);
        }
        Ok(Some(complete))
    }

    pub(super) fn missing_required_rows_for_row_hf_indices(
        &self,
        row_positions: &[i64],
        row_hf_indices: &[Vec<u32>],
        required_hf_indices: &[u32],
    ) -> Result<BTreeMap<u32, Vec<i64>>> {
        let mut missing = BTreeMap::new();
        for (hf_index, positions) in
            required_positions_by_hf_index(row_positions, row_hf_indices, required_hf_indices)?
        {
            let Some(cached) = self.frames.get(&hf_index) else {
                missing.insert(hf_index, positions);
                continue;
            };
            let missing_positions = positions
                .iter()
                .copied()
                .filter_map(|position| {
                    let position_u32 = u32::try_from(position).ok()?;
                    (!cached.rows.contains_key(&position_u32)).then_some(position)
                })
                .collect::<Vec<_>>();
            if !missing_positions.is_empty() {
                missing.insert(hf_index, missing_positions);
            }
        }
        Ok(missing)
    }

    pub(super) fn has_rows_for_hf_indices(&self, position: usize, hf_indices: &[u32]) -> bool {
        let Ok(position) = u32::try_from(position) else {
            return false;
        };
        hf_indices.iter().all(|hf_index| {
            self.frames
                .get(hf_index)
                .is_some_and(|cached| cached.rows.contains_key(&position))
        })
    }

    pub(super) fn frame_for_positions(
        &self,
        hf_index: u32,
        row_positions: &[i64],
    ) -> Result<Option<ActivationFrame>> {
        let Some(cached) = self.frames.get(&hf_index) else {
            return Ok(None);
        };
        let positions = row_positions
            .iter()
            .copied()
            .map(|position| {
                u32::try_from(position)
                    .with_context(|| format!("negative SPD inline tap position {position}"))
            })
            .collect::<Result<Vec<_>>>()?;
        if positions
            .iter()
            .any(|position| !cached.rows.contains_key(position))
        {
            return Ok(None);
        }
        let token_count = positions
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .context("SPD inline tap synthetic token_count overflow")?;
        let row_bytes = self
            .hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .context("SPD inline tap row byte width overflow")?;
        let total_bytes = usize::try_from(token_count)
            .context("SPD inline tap token_count exceeds usize")?
            .checked_mul(row_bytes)
            .context("SPD inline tap synthetic payload overflow")?;
        let mut payload = vec![0_u8; total_bytes];
        for position in positions {
            let row = cached
                .rows
                .get(&position)
                .context("missing cached SPD row after completeness check")?;
            let offset = usize::try_from(position)
                .context("SPD inline tap position exceeds usize")?
                .checked_mul(row_bytes)
                .context("SPD inline tap synthetic offset overflow")?;
            payload[offset..offset + row_bytes].copy_from_slice(row);
        }
        let mut desc = cached.desc;
        desc.token_count = token_count;
        desc.sequence_count = if token_count == 0 { 0 } else { 1 };
        desc.payload_bytes =
            u64::try_from(payload.len()).context("SPD inline tap payload bytes exceed u64")?;
        Ok(Some(ActivationFrame { desc, payload }))
    }
}

fn required_positions_by_hf_index(
    row_positions: &[i64],
    row_hf_indices: &[Vec<u32>],
    required_hf_indices: &[u32],
) -> Result<BTreeMap<u32, Vec<i64>>> {
    if row_positions.len() != row_hf_indices.len() {
        bail!(
            "SPD row metadata length mismatch: positions {}, hf rows {}",
            row_positions.len(),
            row_hf_indices.len()
        );
    }
    let required = required_hf_indices.iter().copied().collect::<BTreeSet<_>>();
    let mut positions = BTreeMap::<u32, BTreeSet<i64>>::new();
    for (position, hf_indices) in row_positions.iter().copied().zip(row_hf_indices) {
        u32::try_from(position)
            .with_context(|| format!("negative SPD inline tap position {position}"))?;
        for hf_index in hf_indices {
            if required.contains(hf_index) {
                positions.entry(*hf_index).or_default().insert(position);
            }
        }
    }
    Ok(positions
        .into_iter()
        .map(|(hf_index, positions)| (hf_index, positions.into_iter().collect()))
        .collect())
}

pub(super) fn common_token_prefix_len(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

pub(super) fn inline_required_hf_indices(required_hf_indices: &[u32]) -> Vec<u32> {
    required_hf_indices
        .iter()
        .copied()
        .filter(|hf_index| *hf_index != 0)
        .collect()
}

pub(super) fn retained_tap_prefix_len_for_context_update(
    previous: &[i32],
    next: &[i32],
    preserve_accepted_extension: bool,
) -> usize {
    if preserve_accepted_extension && next.starts_with(previous) {
        next.len()
    } else {
        common_token_prefix_len(previous, next)
    }
}

#[derive(Clone)]
struct SpdCachedTapFrame {
    desc: skippy_runtime::ActivationDesc,
    rows: BTreeMap<u32, Vec<u8>>,
}

impl SpdCachedTapFrame {
    pub(super) fn new(desc: skippy_runtime::ActivationDesc) -> Self {
        Self {
            desc,
            rows: BTreeMap::new(),
        }
    }
}
