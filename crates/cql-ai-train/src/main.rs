use anyhow::Result;
use burn::prelude::*;
use clap::{Parser, Subcommand, ValueEnum};
use cql_ai_model::{LABELS, io, model::ModelConfig};
use cql_ai_train::synth::generate;
use cql_ai_train::train::{TrainOptions, evaluate, train};
use std::path::PathBuf;

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    /// Pure-Rust CPU backend (burn-flex).
    Flex,
    /// burn's default device: the best enabled backend (needs `--features wgpu` for GPU).
    Auto,
}

#[derive(Parser)]
#[command(
    version,
    about = "Synthetic data + training for the cql-ai anti-pattern Transformer"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate labelled CQL and print label statistics (optionally dump JSONL).
    Generate {
        #[arg(long, default_value_t = 100_000)]
        count: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Number of random schemas the statements are drawn from.
        #[arg(long, default_value_t = 400)]
        schemas: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Train the model and write `model.safetensors` + `config.json` into --out.
    Train {
        #[arg(long, default_value_t = 100_000)]
        train_count: usize,
        #[arg(long, default_value_t = 5_000)]
        val_count: usize,
        #[arg(long, default_value_t = 6)]
        epochs: usize,
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, value_enum, default_value_t = Backend::Auto)]
        backend: Backend,
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,
    },
    /// Evaluate saved weights on freshly generated, unseen schemas.
    Eval {
        #[arg(long, default_value = "artifacts")]
        model: PathBuf,
        #[arg(long, default_value_t = 10_000)]
        count: usize,
        #[arg(long, default_value_t = 999)]
        seed: u64,
    },
}

fn device(b: Backend) -> Device {
    match b {
        Backend::Flex => Device::flex(),
        Backend::Auto => Device::default(),
    }
}

fn print_counts(ds: &cql_ai_train::synth::Dataset) {
    println!("{} samples; positives per label:", ds.len());
    for (l, c) in LABELS.iter().zip(ds.counts()) {
        println!(
            "  {l:<28} {c:>7} ({:.1}%)",
            100.0 * c as f64 / ds.len() as f64
        );
    }
    if !ds.rejected.is_empty() {
        println!("  rejected while generating: {:?}", ds.rejected);
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Generate {
            count,
            seed,
            schemas,
            out,
        } => {
            let ds = generate(count, schemas, seed);
            print_counts(&ds);
            if let Some(p) = out {
                std::fs::write(p, ds.to_jsonl())?;
            }
        }
        Cmd::Train {
            train_count,
            val_count,
            epochs,
            batch_size,
            lr,
            seed,
            backend,
            out,
        } => {
            let dev = device(backend);
            println!("device: {dev:?}");
            let tr = generate(train_count, 600, seed);
            // Different seed => different schemas: validation measures generalisation to unseen schemas.
            let va = generate(val_count, 100, seed + 10_000);
            print_counts(&tr);
            let cfg = ModelConfig::new();
            let opts = TrainOptions {
                epochs,
                batch_size,
                lr,
                weight_decay: 0.01,
                seed,
            };
            let model = train(&cfg, &opts, &tr, &va, &dev)?;
            println!(
                "final validation:\n{}",
                evaluate(&model, &va, &dev, 512).report()
            );
            io::save(&model, &cfg, &out)?;
            println!("saved {}/model.safetensors", out.display());
        }
        Cmd::Eval { model, count, seed } => {
            let dev = Device::default();
            let (m, _) = io::load(&model, &dev)?;
            let ds = generate(count, 100, seed);
            println!("{}", evaluate(&m, &ds, &dev, 512).report());
        }
    }
    Ok(())
}
