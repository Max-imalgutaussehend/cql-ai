//! Loading the measured dataset and turning samples into model inputs.

use crate::labels::{Thresholds, cost_target, label};
use anyhow::{Context, Result};
use cql_ai_model::features::{DataStats, Facts, N_CTX, context, facts};
use cql_ai_model::tokenizer::{SchemaInfo, encode};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// One line of `cql-hpc-collect` output.
#[derive(Clone, Debug, Deserialize)]
pub struct Raw {
    pub world: String,
    pub family: String,
    pub engine: String,
    pub ddl: String,
    pub statement: String,
    pub kind: String,
    pub partitions: usize,
    pub rows_per_partition: usize,
    pub row_bytes: usize,
    pub table_rows: usize,
    pub tombstones: bool,
    pub indexed: bool,
    pub write: bool,
    pub lwt: bool,
    pub limit: Option<u32>,
    pub in_len: usize,
    pub batch_n: usize,
    pub partitions_touched: usize,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub rows_returned: f64,
    pub bytes_est: f64,
    pub base_read_us: f64,
    pub base_write_us: f64,
    pub twin_p50_us: Option<f64>,
    pub limit1_p50_us: Option<f64>,
    pub hot_ratio: Option<f64>,
}

/// A model-ready example.
#[derive(Clone, Debug)]
pub struct Example {
    pub world: String,
    pub family: String,
    pub engine: String,
    pub kind: String,
    pub statement: String,
    pub ids: Vec<u32>,
    /// Context with the data-shape block (known stats).
    pub ctx: [f32; N_CTX],
    /// Context as if no statistics were available (CI without a stats file).
    pub ctx_blind: [f32; N_CTX],
    pub labels: [bool; cql_ai_model::N_LABELS],
    /// `log10(cost / baseline)`.
    pub cost: f32,
}

pub fn read_jsonl(path: &Path) -> Result<Vec<Raw>> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1))
        })
        .collect()
}

/// Encodes raw samples. `normalise` rewrites nothing: statements are already table-name `t`, keyspace-less.
pub fn build(raws: &[Raw], t: &Thresholds) -> Vec<Example> {
    let mut schemas: HashMap<&str, SchemaInfo> = HashMap::new();
    let mut out = Vec::with_capacity(raws.len());
    for r in raws {
        if !(r.p50_us.is_finite() && r.base_read_us.is_finite()
            || r.write && r.base_write_us.is_finite())
        {
            continue;
        }
        let schema = schemas
            .entry(r.ddl.as_str())
            .or_insert_with(|| SchemaInfo::from_ddl(&r.ddl));
        let stats = DataStats {
            partitions: Some(r.partitions as f64),
            rows_per_partition: Some(r.rows_per_partition as f64),
            row_bytes: Some(r.row_bytes as f64),
            table_rows: Some(r.table_rows as f64),
            tombstones: r.tombstones,
            indexed: r.indexed,
        };
        let f = Facts {
            batch_n: r.batch_n,
            ..facts(&r.statement)
        };
        out.push(Example {
            world: r.world.clone(),
            family: r.family.clone(),
            engine: r.engine.clone(),
            kind: r.kind.clone(),
            statement: r.statement.clone(),
            ids: encode(&r.statement, schema),
            ctx: context(Some(&stats), &f),
            ctx_blind: context(None, &f),
            labels: label(r, t),
            cost: cost_target(r),
        });
    }
    out
}
