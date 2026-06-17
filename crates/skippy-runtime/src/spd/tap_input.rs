use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    thread,
};

use anyhow::{Context, Result, bail};

use super::{SpdHeadManifest, SpdHeadTopology, SpdSafetensorsFile};

const PARALLEL_TAP_LINEAR_MIN_DOT_OPS: usize = 2_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct SpdTapInputProjection {
    pub stage_id: u32,
    pub projection_name: String,
    pub hf_indices: Vec<u32>,
    pub projected: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdTapInputFixtureParity {
    pub rows: Vec<SpdTapInputFixtureRowParity>,
    pub max_abs_diff: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdTapInputFixtureRowParity {
    pub row_index: usize,
    pub position_id: i64,
    pub stage_id: u32,
    pub projection_name: String,
    pub hf_indices: Vec<u32>,
    pub max_abs_diff: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TapProjectionSpec {
    name: String,
    hf_indices: Vec<u32>,
    input_width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TapProjectionKey {
    stage_id: u32,
    hf_indices: Vec<u32>,
}

#[derive(Debug, Clone)]
struct CachedTapProjection {
    name: String,
    hf_indices: Vec<u32>,
    input_width: usize,
    weight: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpdTapInputProjector {
    hidden_size: usize,
    projections: BTreeMap<TapProjectionKey, CachedTapProjection>,
}

impl SpdTapInputProjector {
    pub fn from_topology(
        topology: &SpdHeadTopology,
        serving_file: &SpdSafetensorsFile,
    ) -> Result<Self> {
        let hidden_size =
            usize::try_from(topology.hidden_size).context("SPD hidden_size too large")?;
        let mut projections = BTreeMap::new();
        for stage_id in 0..=topology.num_stages {
            let hf_indices = spd_hf_indices_for_stage_id(topology, stage_id)?;
            cache_tap_projection(
                topology,
                serving_file,
                stage_id,
                &hf_indices,
                &mut projections,
            )?;
        }
        Ok(Self {
            hidden_size,
            projections,
        })
    }

    pub fn from_rows(
        topology: &SpdHeadTopology,
        serving_file: &SpdSafetensorsFile,
        row_stage_ids: &[i64],
        row_hf_indices: &[Vec<u32>],
    ) -> Result<Self> {
        if row_stage_ids.len() != row_hf_indices.len() {
            bail!(
                "SPD tap projector row metadata length mismatch: stages {}, hf rows {}",
                row_stage_ids.len(),
                row_hf_indices.len()
            );
        }
        let hidden_size =
            usize::try_from(topology.hidden_size).context("SPD hidden_size too large")?;
        let mut projections = BTreeMap::new();
        for (row_index, hf_indices) in row_hf_indices.iter().enumerate() {
            let stage_id = u32::try_from(row_stage_ids[row_index])
                .with_context(|| format!("SPD fixture row {row_index} has negative stage id"))?;
            cache_tap_projection(
                topology,
                serving_file,
                stage_id,
                hf_indices,
                &mut projections,
            )?;
        }
        Ok(Self {
            hidden_size,
            projections,
        })
    }

    pub fn project(
        &self,
        stage_id: u32,
        hf_indices: &[u32],
        concat_hidden: &[f32],
    ) -> Result<SpdTapInputProjection> {
        let key = TapProjectionKey {
            stage_id,
            hf_indices: hf_indices.to_vec(),
        };
        let projection = self.projections.get(&key).with_context(|| {
            format!("missing cached SPD tap projection for stage {stage_id} {hf_indices:?}")
        })?;
        project_cached_tap_input(stage_id, self.hidden_size, projection, concat_hidden)
    }
}

fn cache_tap_projection(
    topology: &SpdHeadTopology,
    serving_file: &SpdSafetensorsFile,
    stage_id: u32,
    hf_indices: &[u32],
    projections: &mut BTreeMap<TapProjectionKey, CachedTapProjection>,
) -> Result<()> {
    let key = TapProjectionKey {
        stage_id,
        hf_indices: hf_indices.to_vec(),
    };
    if projections.contains_key(&key) {
        return Ok(());
    }
    let spec = tap_projection_spec(topology, stage_id, hf_indices)?;
    let weight = serving_file.read_tensor_f32(&spec.name)?;
    projections.insert(
        key,
        CachedTapProjection {
            name: spec.name,
            hf_indices: spec.hf_indices,
            input_width: spec.input_width,
            weight,
        },
    );
    Ok(())
}

pub fn project_spd_tap_input_row(
    topology: &SpdHeadTopology,
    serving_file: &SpdSafetensorsFile,
    stage_id: u32,
    hf_indices: &[u32],
    concat_hidden: &[f32],
) -> Result<SpdTapInputProjection> {
    let spec = tap_projection_spec(topology, stage_id, hf_indices)?;
    let weight = serving_file.read_tensor_f32(&spec.name)?;
    let hidden_size = usize::try_from(topology.hidden_size).context("SPD hidden_size too large")?;
    let cached = CachedTapProjection {
        name: spec.name,
        hf_indices: spec.hf_indices,
        input_width: spec.input_width,
        weight,
    };
    project_cached_tap_input(stage_id, hidden_size, &cached, concat_hidden)
}

fn project_cached_tap_input(
    stage_id: u32,
    hidden_size: usize,
    projection: &CachedTapProjection,
    concat_hidden: &[f32],
) -> Result<SpdTapInputProjection> {
    let mut projected = vec![0.0; hidden_size];
    linear_into(
        &projection.weight,
        projection.input_width,
        concat_hidden,
        &mut projected,
    )?;
    Ok(SpdTapInputProjection {
        stage_id,
        projection_name: projection.name.clone(),
        hf_indices: projection.hf_indices.clone(),
        projected,
    })
}

pub fn run_spd_tap_input_fixture_parity(
    manifest_path: impl AsRef<Path>,
    fixture_path: impl AsRef<Path>,
) -> Result<SpdTapInputFixtureParity> {
    let manifest_path = manifest_path.as_ref();
    let fixture_path = fixture_path.as_ref();
    let manifest = SpdHeadManifest::from_path(manifest_path)?;
    manifest.ensure_serving_checkpoint_for_runtime(manifest_path)?;
    let serving_file = SpdSafetensorsFile::open(manifest.serving_checkpoint_path(manifest_path)?)?;
    let fixture_file = SpdSafetensorsFile::open(fixture_path)?;
    let cur_shape = &fixture_file.index.tensor("cur_in")?.shape;
    let hidden_size = usize::try_from(manifest.topology.hidden_size)
        .context("SPD fixture hidden_size too large")?;
    if cur_shape.len() != 3 || cur_shape[0] != 1 || cur_shape[2] != hidden_size as u64 {
        bail!(
            "SPD fixture cur_in shape {:?} is not [1, seq, hidden]",
            cur_shape
        );
    }
    let row_count =
        usize::try_from(cur_shape[1]).context("SPD fixture sequence length too large")?;
    let row_i_stages = fixture_file.read_tensor_i64("row_i_stages")?;
    let row_positions = fixture_file.read_tensor_i64("row_positions")?;
    if row_i_stages.len() != row_count || row_positions.len() != row_count {
        bail!(
            "SPD fixture row metadata lengths must match cur_in rows: stages {}, positions {}, rows {}",
            row_i_stages.len(),
            row_positions.len(),
            row_count
        );
    }

    let cur_in = fixture_file.read_tensor_f32("cur_in")?;
    let mut rows = Vec::with_capacity(row_count);
    let mut max_diff = 0.0_f32;
    for row_index in 0..row_count {
        let stage_id = u32::try_from(row_i_stages[row_index])
            .with_context(|| format!("SPD fixture row {row_index} has negative stage id"))?;
        let hf_indices = read_row_hf_indices(&fixture_file, row_index)?;
        let concat_hidden = fixture_file.read_tensor_f32(&format!("tap_row_{row_index}_concat"))?;
        let projection = project_spd_tap_input_row(
            &manifest.topology,
            &serving_file,
            stage_id,
            &hf_indices,
            &concat_hidden,
        )?;
        let expected = row(&cur_in, row_index, hidden_size);
        let row_diff = max_abs_diff(&projection.projected, expected)?;
        max_diff = max_diff.max(row_diff);
        rows.push(SpdTapInputFixtureRowParity {
            row_index,
            position_id: row_positions[row_index],
            stage_id,
            projection_name: projection.projection_name,
            hf_indices: projection.hf_indices,
            max_abs_diff: row_diff,
        });
    }
    Ok(SpdTapInputFixtureParity {
        rows,
        max_abs_diff: max_diff,
    })
}

fn tap_projection_spec(
    topology: &SpdHeadTopology,
    stage_id: u32,
    actual_hf_indices: &[u32],
) -> Result<TapProjectionSpec> {
    let hidden_size = usize::try_from(topology.hidden_size).context("SPD hidden_size too large")?;
    if stage_id == 0 {
        ensure_hf_indices("g0_proj.weight", actual_hf_indices, &[0])?;
        return Ok(TapProjectionSpec {
            name: "g0_proj.weight".to_string(),
            hf_indices: vec![0],
            input_width: hidden_size,
        });
    }
    if stage_id > topology.num_stages {
        bail!(
            "SPD tap input stage_id {} exceeds num_stages {}",
            stage_id,
            topology.num_stages
        );
    }
    let block = usize::try_from(topology.num_stages - stage_id)
        .context("SPD projection block index too large")?;
    let expected = topology
        .shallow_hidden_layer_indices
        .get(block)
        .with_context(|| format!("SPD tap input missing shallow indices for block {block}"))?;
    let name = format!("stage_projs.{block}.weight");
    ensure_hf_indices(&name, actual_hf_indices, expected)?;
    let input_width = hidden_size
        .checked_mul(expected.len())
        .context("SPD tap input projection width overflow")?;
    Ok(TapProjectionSpec {
        name,
        hf_indices: expected.clone(),
        input_width,
    })
}

pub fn spd_hf_indices_for_stage_id(topology: &SpdHeadTopology, stage_id: u32) -> Result<Vec<u32>> {
    if stage_id == 0 {
        return Ok(vec![0]);
    }
    if stage_id > topology.num_stages {
        bail!(
            "SPD stage_id {} exceeds num_stages {}",
            stage_id,
            topology.num_stages
        );
    }
    let block = usize::try_from(topology.num_stages - stage_id)
        .context("SPD projection block index too large")?;
    topology
        .shallow_hidden_layer_indices
        .get(block)
        .cloned()
        .with_context(|| format!("SPD tap input missing shallow indices for block {block}"))
}

pub fn required_spd_hf_indices_for_topology(topology: &SpdHeadTopology) -> Vec<u32> {
    (0..=topology.num_stages)
        .filter_map(|stage_id| spd_hf_indices_for_stage_id(topology, stage_id).ok())
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn ensure_hf_indices(label: &str, actual: &[u32], expected: &[u32]) -> Result<()> {
    if actual != expected {
        bail!(
            "SPD tap input {label} hf_indices mismatch: expected {:?}, got {:?}",
            expected,
            actual
        );
    }
    Ok(())
}

fn read_row_hf_indices(file: &SpdSafetensorsFile, row_index: usize) -> Result<Vec<u32>> {
    file.read_tensor_i64(&format!("tap_row_{row_index}_hf_indices"))?
        .into_iter()
        .map(|value| {
            u32::try_from(value)
                .with_context(|| format!("SPD fixture row {row_index} has negative hf index"))
        })
        .collect()
}

fn row(values: &[f32], row_idx: usize, width: usize) -> &[f32] {
    &values[row_idx * width..(row_idx + 1) * width]
}

fn linear_into(
    weight: &[f32],
    input_width: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    if input.len() != input_width {
        bail!(
            "SPD tap input width mismatch: expected {}, got {}",
            input_width,
            input.len()
        );
    }
    if weight.len() != output.len() * input_width {
        bail!(
            "SPD tap input weight shape mismatch: weight len {}, output {}, input {}",
            weight.len(),
            output.len(),
            input_width
        );
    }
    if should_parallelize_linear(output.len(), input_width) {
        parallel_linear_into(weight, input_width, input, output);
        return Ok(());
    }
    serial_linear_into(weight, input_width, input, output);
    Ok(())
}

fn should_parallelize_linear(output_width: usize, input_width: usize) -> bool {
    thread::available_parallelism().is_ok_and(|parallelism| parallelism.get() > 1)
        && output_width.saturating_mul(input_width) >= PARALLEL_TAP_LINEAR_MIN_DOT_OPS
}

fn serial_linear_into(weight: &[f32], input_width: usize, input: &[f32], output: &mut [f32]) {
    for (out_idx, out) in output.iter_mut().enumerate() {
        let weight_row = &weight[out_idx * input_width..(out_idx + 1) * input_width];
        *out = round_to_bf16(dot(weight_row, input));
    }
}

fn parallel_linear_into(weight: &[f32], input_width: usize, input: &[f32], output: &mut [f32]) {
    let workers = thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .min(output.len());
    let rows_per_worker = output.len().div_ceil(workers);
    thread::scope(|scope| {
        for (chunk_idx, output_chunk) in output.chunks_mut(rows_per_worker).enumerate() {
            let first_row = chunk_idx * rows_per_worker;
            let weight_start = first_row * input_width;
            let weight_end = weight_start + output_chunk.len() * input_width;
            let weight_chunk = &weight[weight_start..weight_end];
            scope.spawn(move || {
                serial_linear_into(weight_chunk, input_width, input, output_chunk);
            });
        }
    });
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "SPD tap input vector length mismatch: {} vs {}",
            left.len(),
            right.len()
        );
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max))
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn round_to_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::spd::{
        SPD_HEAD_MANIFEST_SCHEMA, SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1,
        TORCH_SPD_HEAD_FORMAT_V10,
    };

    #[test]
    fn projects_deep_stage_tap_row_with_expected_block_mapping() {
        let temp = tempfile::tempdir().unwrap();
        let serving_path = temp.path().join("serving.safetensors");
        write_safetensors(
            &serving_path,
            &[
                tensor_f32(
                    "stage_projs.0.weight",
                    &[2, 4],
                    &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                ),
                tensor_f32("stage_projs.1.weight", &[2, 2], &[2.0, 0.0, 0.0, 3.0]),
                tensor_f32("g0_proj.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
            ],
        );
        let serving_file = SpdSafetensorsFile::open(&serving_path).unwrap();
        let topology = test_topology();

        let projection =
            project_spd_tap_input_row(&topology, &serving_file, 2, &[0, 1], &[1.0, 2.0, 3.0, 4.0])
                .unwrap();

        assert_eq!(projection.projection_name, "stage_projs.0.weight");
        assert_eq!(projection.hf_indices, vec![0, 1]);
        assert_eq!(projection.projected, vec![4.0, 6.0]);
    }

    #[test]
    fn cached_projector_matches_direct_tap_projection() {
        let temp = tempfile::tempdir().unwrap();
        let serving_path = temp.path().join("serving.safetensors");
        write_safetensors(
            &serving_path,
            &[
                tensor_f32(
                    "stage_projs.0.weight",
                    &[2, 4],
                    &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                ),
                tensor_f32("stage_projs.1.weight", &[2, 2], &[2.0, 0.0, 0.0, 3.0]),
                tensor_f32("g0_proj.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
            ],
        );
        let serving_file = SpdSafetensorsFile::open(&serving_path).unwrap();
        let topology = test_topology();
        let projector = SpdTapInputProjector::from_rows(
            &topology,
            &serving_file,
            &[2, 2, 1],
            &[vec![0, 1], vec![0, 1], vec![0]],
        )
        .unwrap();
        let concat_hidden = [1.0, 2.0, 3.0, 4.0];

        let direct =
            project_spd_tap_input_row(&topology, &serving_file, 2, &[0, 1], &concat_hidden)
                .unwrap();
        let cached = projector.project(2, &[0, 1], &concat_hidden).unwrap();

        assert_eq!(cached, direct);
    }

    #[test]
    fn topology_required_indices_include_every_stage_role() {
        let topology = SpdHeadTopology {
            hidden_size: 2,
            vocab_size: 8,
            draft_vocab_size: 2,
            num_stages: 4,
            stage_layer_boundaries: Some(vec![8, 16, 24, 32]),
            num_spec_layers: 1,
            trained_with_use_deepest: true,
            shallow_hidden_layer_indices: vec![
                vec![0, 10, 20, 31],
                vec![0, 8, 16, 24],
                vec![0, 8, 16],
                vec![0, 8],
            ],
            spec_init_from_base_layers: None,
            draft_token_ids: None,
        };

        assert_eq!(
            required_spd_hf_indices_for_topology(&topology),
            vec![0, 8, 10, 16, 20, 24, 31]
        );
        assert_eq!(
            spd_hf_indices_for_stage_id(&topology, 3).unwrap(),
            vec![0, 8, 16, 24]
        );
        assert_eq!(spd_hf_indices_for_stage_id(&topology, 0).unwrap(), vec![0]);
    }

    #[test]
    fn topology_projector_preloads_every_stage_role() {
        let temp = tempfile::tempdir().unwrap();
        let serving_path = temp.path().join("serving.safetensors");
        write_safetensors(
            &serving_path,
            &[
                tensor_f32(
                    "stage_projs.0.weight",
                    &[2, 4],
                    &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                ),
                tensor_f32("stage_projs.1.weight", &[2, 2], &[2.0, 0.0, 0.0, 3.0]),
                tensor_f32("g0_proj.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
            ],
        );
        let serving_file = SpdSafetensorsFile::open(&serving_path).unwrap();
        let topology = test_topology();
        let projector = SpdTapInputProjector::from_topology(&topology, &serving_file).unwrap();

        assert_eq!(
            projector
                .project(2, &[0, 1], &[1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .projected,
            vec![4.0, 6.0]
        );
        assert_eq!(
            projector.project(1, &[0], &[5.0, 7.0]).unwrap().projected,
            vec![10.0, 21.0]
        );
        assert_eq!(
            projector.project(0, &[0], &[2.0, 1.0]).unwrap().projected,
            vec![3.0, 1.0]
        );
    }

    #[test]
    fn fixture_parity_reconstructs_cur_in_from_tap_rows() {
        let temp = tempfile::tempdir().unwrap();
        let serving_path = temp.path().join("spd-head.safetensors");
        write_safetensors(
            &serving_path,
            &[
                tensor_f32(
                    "stage_projs.0.weight",
                    &[2, 4],
                    &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                ),
                tensor_f32("stage_projs.1.weight", &[2, 2], &[2.0, 0.0, 0.0, 3.0]),
                tensor_f32("g0_proj.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
                tensor_f32("lm_head.weight", &[2, 2], &[0.0, 0.0, 0.0, 0.0]),
                tensor_f32("spec_layers.0.input_layernorm.weight", &[2], &[1.0, 1.0]),
                tensor_f32(
                    "spec_layers.0.post_attention_layernorm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
            ],
        );
        let manifest_path = write_manifest(temp.path(), &serving_path);
        let fixture_path = temp.path().join("fixture.safetensors");
        write_safetensors(
            &fixture_path,
            &[
                tensor_f32("cur_in", &[1, 3, 2], &[4.0, 6.0, 10.0, 21.0, 3.0, 1.0]),
                tensor_i64("row_i_stages", &[3], &[2, 1, 0]),
                tensor_i64("row_positions", &[3], &[10, 11, 12]),
                tensor_f32("tap_row_0_concat", &[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]),
                tensor_i64("tap_row_0_hf_indices", &[2], &[0, 1]),
                tensor_f32("tap_row_1_concat", &[1, 1, 2], &[5.0, 7.0]),
                tensor_i64("tap_row_1_hf_indices", &[1], &[0]),
                tensor_f32("tap_row_2_concat", &[1, 1, 2], &[2.0, 1.0]),
                tensor_i64("tap_row_2_hf_indices", &[1], &[0]),
            ],
        );

        let parity = run_spd_tap_input_fixture_parity(manifest_path, fixture_path).unwrap();

        assert_eq!(parity.max_abs_diff, 0.0);
        assert_eq!(parity.rows.len(), 3);
        assert_eq!(parity.rows[0].projection_name, "stage_projs.0.weight");
        assert_eq!(parity.rows[2].projection_name, "g0_proj.weight");
    }

    fn test_topology() -> SpdHeadTopology {
        SpdHeadTopology {
            hidden_size: 2,
            vocab_size: 8,
            draft_vocab_size: 2,
            num_stages: 2,
            stage_layer_boundaries: Some(vec![1, 2]),
            num_spec_layers: 1,
            trained_with_use_deepest: true,
            shallow_hidden_layer_indices: vec![vec![0, 1], vec![0]],
            spec_init_from_base_layers: None,
            draft_token_ids: None,
        }
    }

    fn write_manifest(temp_path: &Path, serving_path: &Path) -> std::path::PathBuf {
        let sha256 = file_sha256(serving_path);
        let manifest = SpdHeadManifest {
            schema: SPD_HEAD_MANIFEST_SCHEMA.to_string(),
            checkpoint: super::super::SpdHeadCheckpoint {
                path: "unused.pt".to_string(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
                bytes: 1,
            },
            serving_checkpoint: Some(super::super::SpdHeadServingCheckpoint {
                path: "spd-head.safetensors".to_string(),
                sha256,
                bytes: fs::metadata(serving_path).unwrap().len(),
                format: SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1.to_string(),
                tensor_count: 6,
                dtype: "F32".to_string(),
            }),
            source: super::super::SpdHeadSource {
                format: TORCH_SPD_HEAD_FORMAT_V10.to_string(),
                reference_repo: None,
                base_model_path: "Qwen/Qwen3-test".to_string(),
                model_type: Some("qwen3".to_string()),
                checkpoint_version: 10,
            },
            topology: test_topology(),
        };
        let manifest_path = temp_path.join("skippy-spd-head.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        manifest_path
    }

    #[derive(Debug)]
    struct TestTensor {
        name: &'static str,
        dtype: &'static str,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn tensor_f32(name: &'static str, shape: &[u64], values: &[f32]) -> TestTensor {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        TestTensor {
            name,
            dtype: "F32",
            shape: shape.to_vec(),
            bytes,
        }
    }

    fn tensor_i64(name: &'static str, shape: &[u64], values: &[i64]) -> TestTensor {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        TestTensor {
            name,
            dtype: "I64",
            shape: shape.to_vec(),
            bytes,
        }
    }

    fn write_safetensors(path: &Path, tensors: &[TestTensor]) {
        let mut header_entries = serde_json::Map::new();
        let mut data = Vec::new();
        for tensor in tensors {
            let start = data.len() as u64;
            data.extend_from_slice(&tensor.bytes);
            let end = data.len() as u64;
            header_entries.insert(
                tensor.name.to_string(),
                serde_json::json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        header_entries.insert("__metadata__".to_string(), serde_json::json!({}));
        let header = serde_json::to_vec(&serde_json::Value::Object(header_entries)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        fs::write(path, bytes).unwrap();
    }

    fn file_sha256(path: &Path) -> String {
        let bytes = fs::read(path).unwrap();
        format!("{:x}", Sha256::digest(bytes))
    }
}
