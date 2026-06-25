use std::{collections::BTreeMap, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::cli::GlmDsaOpReportArgs;

const OP_TIMING_PREFIX: &str = "skippy: glm_dsa_op_timing ";
const SIDEBAND_PREFIX: &str = "skippy: glm_dsa_top_k_sideband_forward ";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Default, Serialize)]
struct OpBucket {
    nodes: u64,
    elapsed_us: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct PhaseSummary {
    records: usize,
    tokens: u64,
    total_us: u64,
    avg_total_us_per_record: Option<f64>,
    avg_total_us_per_token: Option<f64>,
    indexer_topk: OpBucket,
    sparse_mask: OpBucket,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_mask_fill: Option<OpBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_mask_topk: Option<OpBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_mask_add: Option<OpBucket>,
    mla_attention: OpBucket,
    routed_moe: OpBucket,
    shared_expert: OpBucket,
}

#[derive(Debug, Clone, Serialize)]
struct LogSummary {
    path: PathBuf,
    records: usize,
    stage_records: BTreeMap<i32, BTreeMap<Phase, PhaseSummary>>,
    sideband_records: BTreeMap<String, BTreeMap<Phase, SidebandSummary>>,
}

#[derive(Debug, Serialize)]
struct GlmDsaOpReport {
    logs: Vec<LogSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimingRecord {
    stage: i32,
    tokens: u64,
    total_us: u64,
    indexer_topk_nodes: u64,
    indexer_topk_us: u64,
    sparse_mask_nodes: u64,
    sparse_mask_us: u64,
    sparse_mask_fill_nodes: Option<u64>,
    sparse_mask_fill_us: Option<u64>,
    sparse_mask_topk_nodes: Option<u64>,
    sparse_mask_topk_us: Option<u64>,
    sparse_mask_add_nodes: Option<u64>,
    sparse_mask_add_us: Option<u64>,
    mla_attention_nodes: u64,
    mla_attention_us: u64,
    routed_moe_nodes: u64,
    routed_moe_us: u64,
    shared_expert_nodes: u64,
    shared_expert_us: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SidebandSummary {
    records: usize,
    tokens: u64,
    hidden_bytes: u64,
    sideband_bytes: u64,
    sideband_i32: u64,
    avg_hidden_bytes_per_token: Option<f64>,
    avg_sideband_bytes_per_token: Option<f64>,
    sideband_to_hidden_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SidebandRecord {
    stage: String,
    kind: String,
    tokens: u64,
    hidden_bytes: u64,
    sideband_bytes: u64,
    sideband_i32: u64,
}

pub fn glm_dsa_op_report(args: GlmDsaOpReportArgs) -> Result<()> {
    let output = args.output.clone();
    let report = build_report(&args)?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn build_report(args: &GlmDsaOpReportArgs) -> Result<GlmDsaOpReport> {
    let mut logs = Vec::with_capacity(args.log.len());
    for path in &args.log {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let records = parse_timing_records(&text)
            .with_context(|| format!("parse GLM-DSA op timing records in {}", path.display()))?;
        if records.is_empty() {
            bail!("{} contains no GLM-DSA op timing records", path.display());
        }
        let sideband_records = parse_sideband_records(&text).with_context(|| {
            format!("parse GLM-DSA top-k sideband records in {}", path.display())
        })?;
        let records = match args.first_records {
            Some(limit) => records.into_iter().take(limit).collect::<Vec<_>>(),
            None => records,
        };
        let sideband_records = match args.first_records {
            Some(limit) => sideband_records.into_iter().take(limit).collect::<Vec<_>>(),
            None => sideband_records,
        };
        logs.push(summarize_log(path.clone(), &records, &sideband_records));
    }
    Ok(GlmDsaOpReport { logs })
}

fn parse_timing_records(text: &str) -> Result<Vec<TimingRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(OP_TIMING_PREFIX)
                .map(|index| &line[index + OP_TIMING_PREFIX.len()..])
        })
        .map(parse_timing_record)
        .collect()
}

fn parse_timing_record(line: &str) -> Result<TimingRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    let sparse_mask_fill = parse_optional_bucket(&fields, "sparse_mask_fill")?;
    let sparse_mask_topk = parse_optional_bucket(&fields, "sparse_mask_topk")?;
    let sparse_mask_add = parse_optional_bucket(&fields, "sparse_mask_add")?;
    Ok(TimingRecord {
        stage: parse_field(&fields, "stage")?,
        tokens: parse_field(&fields, "tokens")?,
        total_us: parse_field(&fields, "total_us")?,
        indexer_topk_nodes: parse_field(&fields, "indexer_topk_nodes")?,
        indexer_topk_us: parse_field(&fields, "indexer_topk_us")?,
        sparse_mask_nodes: parse_field(&fields, "sparse_mask_nodes")?,
        sparse_mask_us: parse_field(&fields, "sparse_mask_us")?,
        sparse_mask_fill_nodes: sparse_mask_fill.nodes,
        sparse_mask_fill_us: sparse_mask_fill.elapsed_us,
        sparse_mask_topk_nodes: sparse_mask_topk.nodes,
        sparse_mask_topk_us: sparse_mask_topk.elapsed_us,
        sparse_mask_add_nodes: sparse_mask_add.nodes,
        sparse_mask_add_us: sparse_mask_add.elapsed_us,
        mla_attention_nodes: parse_field(&fields, "mla_attention_nodes")?,
        mla_attention_us: parse_field(&fields, "mla_attention_us")?,
        routed_moe_nodes: parse_field(&fields, "routed_moe_nodes")?,
        routed_moe_us: parse_field(&fields, "routed_moe_us")?,
        shared_expert_nodes: parse_field(&fields, "shared_expert_nodes")?,
        shared_expert_us: parse_field(&fields, "shared_expert_us")?,
    })
}

#[derive(Debug, Clone, Copy)]
struct OptionalBucketFields {
    nodes: Option<u64>,
    elapsed_us: Option<u64>,
}

fn parse_optional_bucket(
    fields: &BTreeMap<&str, &str>,
    name: &str,
) -> Result<OptionalBucketFields> {
    let nodes = parse_optional_field(fields, &format!("{name}_nodes"))?;
    let elapsed_us = parse_optional_field(fields, &format!("{name}_us"))?;
    if nodes.is_some() != elapsed_us.is_some() {
        bail!("{name} must include both nodes and us fields");
    }
    Ok(OptionalBucketFields { nodes, elapsed_us })
}

fn parse_sideband_records(text: &str) -> Result<Vec<SidebandRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(SIDEBAND_PREFIX)
                .map(|index| &line[index + SIDEBAND_PREFIX.len()..])
        })
        .map(parse_sideband_record)
        .collect()
}

fn parse_sideband_record(line: &str) -> Result<SidebandRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(SidebandRecord {
        stage: parse_string_field(&fields, "stage")?,
        kind: parse_string_field(&fields, "kind")?,
        tokens: parse_field(&fields, "tokens")?,
        hidden_bytes: parse_field(&fields, "hidden_bytes")?,
        sideband_bytes: parse_field(&fields, "sideband_bytes")?,
        sideband_i32: parse_field(&fields, "sideband_i32")?,
    })
}

fn parse_field<T>(fields: &BTreeMap<&str, &str>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    fields
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
}

fn parse_optional_field<T>(fields: &BTreeMap<&str, &str>, name: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    fields
        .get(name)
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
        })
        .transpose()
}

fn parse_string_field(fields: &BTreeMap<&str, &str>, name: &str) -> Result<String> {
    Ok(fields
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .to_string())
}

fn summarize_log(
    path: PathBuf,
    records: &[TimingRecord],
    sideband_records: &[SidebandRecord],
) -> LogSummary {
    let mut stage_records: BTreeMap<i32, BTreeMap<Phase, PhaseSummary>> = BTreeMap::new();
    for record in records {
        let phase = if record.tokens == 1 {
            Phase::Decode
        } else {
            Phase::Prefill
        };
        let summary = stage_records
            .entry(record.stage)
            .or_default()
            .entry(phase)
            .or_default();
        summary.records += 1;
        summary.tokens += record.tokens;
        summary.total_us += record.total_us;
        add_bucket(
            &mut summary.indexer_topk,
            record.indexer_topk_nodes,
            record.indexer_topk_us,
        );
        add_bucket(
            &mut summary.sparse_mask,
            record.sparse_mask_nodes,
            record.sparse_mask_us,
        );
        add_optional_bucket(
            &mut summary.sparse_mask_fill,
            record.sparse_mask_fill_nodes,
            record.sparse_mask_fill_us,
        );
        add_optional_bucket(
            &mut summary.sparse_mask_topk,
            record.sparse_mask_topk_nodes,
            record.sparse_mask_topk_us,
        );
        add_optional_bucket(
            &mut summary.sparse_mask_add,
            record.sparse_mask_add_nodes,
            record.sparse_mask_add_us,
        );
        add_bucket(
            &mut summary.mla_attention,
            record.mla_attention_nodes,
            record.mla_attention_us,
        );
        add_bucket(
            &mut summary.routed_moe,
            record.routed_moe_nodes,
            record.routed_moe_us,
        );
        add_bucket(
            &mut summary.shared_expert,
            record.shared_expert_nodes,
            record.shared_expert_us,
        );
    }
    for phases in stage_records.values_mut() {
        for summary in phases.values_mut() {
            summary.avg_total_us_per_record = nonzero_div(summary.total_us, summary.records as u64);
            summary.avg_total_us_per_token = nonzero_div(summary.total_us, summary.tokens);
        }
    }
    let sideband_records = summarize_sideband_records(sideband_records);
    LogSummary {
        path,
        records: records.len(),
        stage_records,
        sideband_records,
    }
}

fn summarize_sideband_records(
    records: &[SidebandRecord],
) -> BTreeMap<String, BTreeMap<Phase, SidebandSummary>> {
    let mut stages: BTreeMap<String, BTreeMap<Phase, SidebandSummary>> = BTreeMap::new();
    for record in records {
        let phase = sideband_phase(&record.kind, record.tokens);
        let summary = stages
            .entry(record.stage.clone())
            .or_default()
            .entry(phase)
            .or_default();
        summary.records += 1;
        summary.tokens += record.tokens;
        summary.hidden_bytes += record.hidden_bytes;
        summary.sideband_bytes += record.sideband_bytes;
        summary.sideband_i32 += record.sideband_i32;
    }
    for phases in stages.values_mut() {
        for summary in phases.values_mut() {
            summary.avg_hidden_bytes_per_token = nonzero_div(summary.hidden_bytes, summary.tokens);
            summary.avg_sideband_bytes_per_token =
                nonzero_div(summary.sideband_bytes, summary.tokens);
            summary.sideband_to_hidden_ratio =
                nonzero_div(summary.sideband_bytes, summary.hidden_bytes);
        }
    }
    stages
}

fn sideband_phase(kind: &str, tokens: u64) -> Phase {
    if kind == "DecodeEmbd" || tokens == 1 {
        Phase::Decode
    } else {
        Phase::Prefill
    }
}

fn add_bucket(bucket: &mut OpBucket, nodes: u64, elapsed_us: u64) {
    bucket.nodes += nodes;
    bucket.elapsed_us += elapsed_us;
}

fn add_optional_bucket(bucket: &mut Option<OpBucket>, nodes: Option<u64>, elapsed_us: Option<u64>) {
    if let (Some(nodes), Some(elapsed_us)) = (nodes, elapsed_us) {
        add_bucket(
            bucket.get_or_insert_with(OpBucket::default),
            nodes,
            elapsed_us,
        );
    }
}

fn nonzero_div(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::{
        Phase, parse_sideband_record, parse_sideband_records, parse_timing_record,
        parse_timing_records, summarize_log,
    };

    const LINE: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_SPARSE_BREAKDOWN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 sparse_mask_fill_nodes=47 sparse_mask_fill_us=1000 sparse_mask_topk_nodes=47 sparse_mask_topk_us=2000 sparse_mask_add_nodes=47 sparse_mask_add_us=3000 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const SIDEBAND_LINE: &str = "skippy: glm_dsa_top_k_sideband_forward stage=stage-0 request=1 session=2 kind=DecodeEmbd pos_start=718 tokens=1 hidden_bytes=24576 sideband_bytes=3072 sideband_i32=768";

    #[test]
    fn parses_timing_record_with_prefix() {
        let records = parse_timing_records(LINE).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].stage, 1);
        assert_eq!(records[0].tokens, 128);
        assert_eq!(records[0].indexer_topk_us, 129065);
        assert_eq!(records[0].shared_expert_nodes, 47);
    }

    #[test]
    fn parses_optional_sparse_mask_breakdown() {
        let record = parse_timing_record(LINE_WITH_SPARSE_BREAKDOWN).unwrap();
        assert_eq!(record.sparse_mask_fill_us, Some(1000));
        assert_eq!(record.sparse_mask_topk_us, Some(2000));
        assert_eq!(record.sparse_mask_add_us, Some(3000));

        let summary = summarize_log("stage1.log".into(), &[record], &[]);
        let prefill = summary
            .stage_records
            .get(&1)
            .unwrap()
            .get(&Phase::Prefill)
            .unwrap();
        assert_eq!(prefill.sparse_mask.elapsed_us, 114543);
        assert_eq!(prefill.sparse_mask_fill.as_ref().unwrap().elapsed_us, 1000);
        assert_eq!(prefill.sparse_mask_topk.as_ref().unwrap().elapsed_us, 2000);
        assert_eq!(prefill.sparse_mask_add.as_ref().unwrap().elapsed_us, 3000);
    }

    #[test]
    fn rejects_partial_sparse_mask_breakdown() {
        let error = parse_timing_record(&LINE.replace(
            "sparse_mask_nodes=235",
            "sparse_mask_nodes=235 sparse_mask_fill_nodes=47",
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("sparse_mask_fill must include both nodes and us fields"));
    }

    #[test]
    fn rejects_missing_fields() {
        let error = parse_timing_record("stage=0 tokens=1")
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing total_us"));
    }

    #[test]
    fn summarizes_prefill_and_decode() {
        let text = format!(
            "{LINE}\n{}",
            LINE.replace("tokens=128", "tokens=1")
                .replace("total_us=1475800", "total_us=200")
        );
        let records = parse_timing_records(&text).unwrap();
        let summary = summarize_log("stage1.log".into(), &records, &[]);
        let stages = summary.stage_records.get(&1).unwrap();
        let prefill = stages.get(&Phase::Prefill).unwrap();
        let decode = stages.get(&Phase::Decode).unwrap();
        assert_eq!(prefill.records, 1);
        assert_eq!(prefill.tokens, 128);
        assert_eq!(decode.records, 1);
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.total_us, 200);
    }

    #[test]
    fn parses_sideband_record_with_prefix() {
        let records = parse_sideband_records(SIDEBAND_LINE).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].stage, "stage-0");
        assert_eq!(records[0].kind, "DecodeEmbd");
        assert_eq!(records[0].sideband_bytes, 3072);
        assert_eq!(records[0].sideband_i32, 768);
    }

    #[test]
    fn rejects_malformed_sideband_record() {
        let error = parse_sideband_record("stage=stage-0 kind=DecodeEmbd")
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing tokens"));
    }

    #[test]
    fn summarizes_sideband_payload_ratios() {
        let timing = parse_timing_records(LINE).unwrap();
        let sideband = parse_sideband_records(SIDEBAND_LINE).unwrap();
        let summary = summarize_log("stage0.log".into(), &timing, &sideband);
        let stages = summary.sideband_records.get("stage-0").unwrap();
        let decode = stages.get(&Phase::Decode).unwrap();
        assert_eq!(decode.records, 1);
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.hidden_bytes, 24576);
        assert_eq!(decode.sideband_bytes, 3072);
        assert_eq!(decode.avg_sideband_bytes_per_token, Some(3072.0));
        assert_eq!(decode.sideband_to_hidden_ratio, Some(0.125));
    }
}
