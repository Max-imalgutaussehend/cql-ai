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
    about = "Detects CQL anti-patterns with a tiny Transformer over cqlc lexer tokens (cqlz family)"
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
        /// Directory with model.safetensors + config.json (default: ~/.cache/cql-ai, fetched from Hugging Face).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Probability at which a finding is reported.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
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
            model,
            threshold,
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
            let report = check::analyze(&files, &schema, &net, &device, threshold)?;
            match format {
                Format::Text => eprint!("{}", check::render(&report)),
                Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
            }
            Ok(deny_warnings && !report.findings.is_empty())
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
