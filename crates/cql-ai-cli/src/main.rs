use anyhow::Result;
use burn::prelude::*;
use clap::{Parser, Subcommand, ValueEnum};
use cql_ai_cli::{cache, check};
use cql_ai_model::io;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Text,
    Json,
}

#[derive(Parser)]
#[command(
    version,
    about = "Predicts CQL performance bottlenecks with a tiny Transformer over cqlc lexer tokens and data-shape statistics (cqlz family)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Check `.cql` files or directories.
    Check {
        /// Files or directories with queries (DDL in the same files is used as the schema).
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Extra schema files or directories (`CREATE TABLE ...`), repeatable.
        #[arg(long)]
        schema: Vec<PathBuf>,
        /// TOML file with `[table]` sections: partitions, rows_per_partition, row_bytes, tombstones, indexed.
        #[arg(long)]
        stats: Vec<PathBuf>,
        /// Directory with model.safetensors + config.json (default: ~/.cache/cql-ai, fetched from Hugging Face).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Probability at which a finding is reported.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        /// Also report statements predicted at least this many times as costly as a basic operation.
        #[arg(long, default_value_t = 20.0)]
        min_cost: f32,
        /// Exit with status 1 when anything is reported.
        #[arg(long)]
        deny_warnings: bool,
        /// Never touch the network.
        #[arg(long)]
        offline: bool,
        /// Re-download the model even if cached.
        #[arg(long)]
        refresh: bool,
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
}

fn run() -> Result<bool> {
    match Cli::parse().cmd {
        Cmd::Check {
            paths,
            schema,
            stats,
            model,
            threshold,
            min_cost,
            deny_warnings,
            offline,
            refresh,
            format,
        } => {
            let dir = cache::resolve_model(model.as_deref(), refresh, offline)?;
            let device = Device::flex();
            let (net, _) = io::load(&dir, &device)?;
            let files = check::collect_files(&paths)?;
            let schema = check::collect_files(&schema)?;
            let report = check::analyze(
                &files,
                &schema,
                &stats,
                &net,
                &device,
                check::Options {
                    threshold,
                    min_cost,
                },
            )?;
            match format {
                Format::Text => eprint!("{}", check::render(&report)),
                Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
            }
            Ok(deny_warnings && !(report.findings.is_empty() && report.slow_statements.is_empty()))
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
