use anyhow::{Result, bail};
use burn::prelude::*;
use clap::{Parser, Subcommand};
use cql_ai_model::{LABELS, io, model::ModelConfig};
use cql_ai_train::baseline::rules;
use cql_ai_train::dataset::{Example, build, read_jsonl};
use cql_ai_train::labels::Thresholds;
use cql_ai_train::train::{self, TrainOptions, evaluate, score};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Train and evaluate the cql-ai performance bottleneck model on measured data"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Label statistics of one or more datasets.
    Stats {
        #[arg(long, required = true)]
        data: Vec<PathBuf>,
    },
    /// Train, evaluate on held-out worlds and held-out families, compare with baselines, save the model.
    Train {
        #[arg(long, required = true)]
        data: Vec<PathBuf>,
        /// Families (e.g. `ycsb,kv_comments`) that are never trained on.
        #[arg(long, value_delimiter = ',', default_value = "ycsb,kv_comments")]
        holdout_families: Vec<String>,
        /// Engine whose samples are used for training (default: all); others are a cross-engine test set.
        #[arg(long)]
        train_engine: Option<String>,
        #[arg(long, default_value_t = 60)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,
    },
}

fn load(paths: &[PathBuf]) -> Result<Vec<Example>> {
    let mut raws = Vec::new();
    for p in paths {
        raws.extend(read_jsonl(p)?);
    }
    Ok(build(&raws, &Thresholds::default()))
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

fn print_stats(ex: &[Example]) {
    let mut by_label = [0usize; cql_ai_model::N_LABELS];
    for e in ex {
        for (c, l) in e.labels.iter().enumerate() {
            by_label[c] += usize::from(*l);
        }
    }
    println!(
        "{} statements, {} worlds, families: {:?}",
        ex.len(),
        ex.iter().map(|e| &e.world).collect::<BTreeSet<_>>().len(),
        ex.iter()
            .map(|e| e.family.as_str())
            .collect::<BTreeSet<_>>()
    );
    for (l, n) in LABELS.iter().zip(by_label) {
        println!(
            "  {l:<16} {n:>5} ({:.1}%)",
            100.0 * n as f64 / ex.len() as f64
        );
    }
    println!(
        "  healthy (no label) {:>4}",
        ex.iter().filter(|e| e.labels.iter().all(|l| !l)).count()
    );
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Stats { data } => print_stats(&load(&data)?),
        Cmd::Train {
            data,
            holdout_families,
            train_engine,
            epochs,
            batch_size,
            lr,
            seed,
            out,
        } => {
            let all = load(&data)?;
            print_stats(&all);
            if all.is_empty() {
                bail!("empty dataset");
            }
            let engine_ok = |e: &Example| train_engine.as_ref().is_none_or(|t| &e.engine == t);
            let held_family = |e: &Example| holdout_families.contains(&e.family);
            // world-level split: ~20% of the remaining worlds of each family are the in-distribution test set
            let mut worlds_by_family: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
            for e in all.iter().filter(|e| engine_ok(e) && !held_family(e)) {
                worlds_by_family
                    .entry(&e.family)
                    .or_default()
                    .insert(&e.world);
            }
            let mut test_worlds = BTreeSet::new();
            let mut val_worlds = BTreeSet::new();
            for ws in worlds_by_family.values() {
                let mut v: Vec<&&str> = ws.iter().collect();
                v.sort_by_key(|w| fnv(&format!("{seed}{w}")));
                let n_test = if v.len() >= 3 {
                    (v.len() as f64 * 0.2).round().max(1.0) as usize
                } else {
                    0
                };
                let n_val = if v.len() >= 5 { 1 } else { 0 };
                test_worlds.extend(v.iter().take(n_test).map(|w| **w));
                val_worlds.extend(v.iter().skip(n_test).take(n_val).map(|w| **w));
            }
            let train_set: Vec<&Example> = all
                .iter()
                .filter(|e| {
                    engine_ok(e)
                        && !held_family(e)
                        && !test_worlds.contains(e.world.as_str())
                        && !val_worlds.contains(e.world.as_str())
                })
                .collect();
            let val_set: Vec<&Example> = all
                .iter()
                .filter(|e| val_worlds.contains(e.world.as_str()))
                .collect();
            let test_world: Vec<&Example> = all
                .iter()
                .filter(|e| test_worlds.contains(e.world.as_str()))
                .collect();
            let test_family: Vec<&Example> = all.iter().filter(|e| held_family(e)).collect();
            let test_engine: Vec<&Example> = all.iter().filter(|e| !engine_ok(e)).collect();
            println!(
                "split: {} train / {} val / {} held-out worlds / {} held-out families / {} other-engine",
                train_set.len(),
                val_set.len(),
                test_world.len(),
                test_family.len(),
                test_engine.len()
            );

            let device = if cfg!(feature = "wgpu") {
                Device::default()
            } else {
                Device::flex()
            };
            let cfg = ModelConfig::new();
            let opts = TrainOptions {
                epochs,
                batch_size,
                lr,
                weight_decay: 0.01,
                seed,
                blind_prob: 0.3,
                cost_weight: 1.0,
            };
            let model = train::train(
                &cfg,
                &opts,
                &train_set,
                if val_set.is_empty() {
                    &test_world
                } else {
                    &val_set
                },
                &device,
            )?;

            for (name, set) in [
                (
                    "held-out WORLDS (same families, unseen data shapes)",
                    &test_world,
                ),
                ("held-out FAMILIES (unseen workloads)", &test_family),
                ("other ENGINE", &test_engine),
            ] {
                if set.is_empty() {
                    continue;
                }
                println!("\n== {name}");
                println!(
                    "  model with data statistics:\n{}",
                    evaluate(&model, set, false, &device).report()
                );
                println!(
                    "  model WITHOUT data statistics (CI without a stats file):\n{}",
                    evaluate(&model, set, true, &device).report()
                );
                let rl: Vec<[bool; cql_ai_model::N_LABELS]> =
                    set.iter().map(|e| rules(e)).collect();
                println!(
                    "  rule baseline (keyword heuristics, no data shape):\n{}",
                    score(set, &rl, &[]).report()
                );
            }
            io::save(&model, &cfg, &out)?;
            println!("saved {}/model.safetensors", out.display());
        }
    }
    Ok(())
}
