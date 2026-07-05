mod chat_corpus;
mod cli;
mod distributed;
mod glm_dsa_layer_microbench;
mod glm_dsa_microbench_summary;
mod glm_dsa_op_report;
mod glm_dsa_report_aggregate;
mod local_single;
mod local_split;
mod model_identity;
mod support;
mod token_lengths;
mod verify_span_binary;
mod verify_span_local;

use anyhow::Result;
use clap::Parser;

use crate::{
    chat_corpus::chat_corpus,
    cli::{Cli, CommandKind},
    distributed::{drive_existing, focused_runtime, run_distributed},
    glm_dsa_layer_microbench::glm_dsa_layer_microbench,
    glm_dsa_op_report::{glm_dsa_op_compare, glm_dsa_op_report},
    glm_dsa_report_aggregate::glm_dsa_aggregate_reports,
    local_single::local_single,
    local_split::{
        local_split_binary, local_split_chain_binary, local_split_chain_inprocess,
        local_split_compare, local_split_inprocess,
    },
    token_lengths::token_lengths,
    verify_span_binary::verify_span_binary,
    verify_span_local::verify_span_local,
};

fn main() -> Result<()> {
    match Cli::parse().command {
        CommandKind::LocalSingle(args) => local_single(args),
        CommandKind::LocalSplitInprocess(args) => local_split_inprocess(args),
        CommandKind::LocalSplitBinary(args) => local_split_binary(args),
        CommandKind::LocalSplitCompare(args) => local_split_compare(args),
        CommandKind::LocalSplitChainInprocess(args) => local_split_chain_inprocess(args),
        CommandKind::LocalSplitChainBinary(args) => local_split_chain_binary(args),
        CommandKind::VerifySpanLocal(args) => verify_span_local(args),
        CommandKind::VerifySpanBinary(args) => verify_span_binary(args),
        CommandKind::ChatCorpus(args) => chat_corpus(args),
        CommandKind::TokenLengths(args) => token_lengths(args),
        CommandKind::FocusedRuntime(args) => focused_runtime(args),
        CommandKind::DriveExisting(args) => drive_existing(args),
        CommandKind::GlmDsaLayerMicrobench(args) => glm_dsa_layer_microbench(args),
        CommandKind::GlmDsaOpReport(args) => glm_dsa_op_report(args),
        CommandKind::GlmDsaOpCompare(args) => glm_dsa_op_compare(args),
        CommandKind::GlmDsaAggregateReports(args) => glm_dsa_aggregate_reports(args),
        CommandKind::Run(args) => run_distributed(args),
    }
}
