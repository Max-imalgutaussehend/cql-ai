//! Collect `.cql` files, predict performance bottlenecks, render findings.

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use cql_ai_model::features::{DataStats, Facts, context, facts};
use cql_ai_model::model::{BottleneckNet, batch_ctx, batch_tokens};
use cql_ai_model::tokenizer::{SchemaInfo, encode, statements};
use cql_ai_model::{LABEL_HELP, LABELS};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use walkdir::WalkDir;

/// One predicted bottleneck.
#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub label: &'static str,
    pub probability: f32,
    /// Predicted cost as a multiple of a point read (reads) or a plain insert (writes).
    pub cost_ratio: f32,
    pub statement: String,
    /// Where the data-shape numbers came from.
    pub data_source: String,
}

#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub files: usize,
    pub statements: usize,
    pub with_data_stats: usize,
    pub findings: Vec<Finding>,
    /// Statements predicted above `--min-cost` with no label (generic slowness).
    pub slow_statements: Vec<Finding>,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub threshold: f32,
    /// Predicted cost ratio above which a statement is reported even without a bottleneck label.
    pub min_cost: f32,
}

/// All `.cql` files below `paths` (files are taken as given), sorted.
pub fn collect_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_file() {
            out.push(p.clone());
            continue;
        }
        if !p.exists() {
            bail!("{} does not exist", p.display());
        }
        out.extend(
            WalkDir::new(p)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_type().is_file() && e.path().extension().is_some_and(|x| x == "cql")
                })
                .map(|e| e.into_path()),
        );
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn first_word(s: &str) -> String {
    s.split(|c: char| !c.is_ascii_alphabetic())
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn parse_number(v: &str) -> Option<f64> {
    v.replace('_', "").parse::<f64>().ok()
}

/// `-- hpc: partitions=100000 rows=10000 row_bytes=300 tombstones=true indexed=true batch=10`
fn parse_hpc_line(line: &str) -> Option<HashMap<String, String>> {
    let rest = line
        .trim_start()
        .strip_prefix("--")?
        .trim_start()
        .strip_prefix("hpc:")?;
    Some(
        rest.split_whitespace()
            .filter_map(|kv| kv.split_once('='))
            .map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))
            .collect(),
    )
}

fn stats_from(kv: &HashMap<String, String>) -> DataStats {
    let num = |names: &[&str]| {
        names
            .iter()
            .find_map(|n| kv.get(*n).and_then(|v| parse_number(v)))
    };
    let flag = |n: &str| {
        kv.get(n)
            .is_some_and(|v| matches!(v.as_str(), "true" | "1" | "yes"))
    };
    DataStats {
        partitions: num(&["partitions"]),
        rows_per_partition: num(&["rows", "rows_per_partition"]),
        row_bytes: num(&["row_bytes"]),
        table_rows: num(&["table_rows"]),
        tombstones: flag("tombstones"),
        indexed: flag("indexed"),
    }
}

/// Table name -> stats from `-- hpc:` comments that precede a `CREATE TABLE`, plus a TOML file of `[table]` sections.
fn collect_stats(
    sources: &[String],
    stats_files: &[PathBuf],
) -> Result<HashMap<String, DataStats>> {
    let mut out = HashMap::new();
    for src in sources {
        let mut pending: Option<DataStats> = None;
        for line in src.lines() {
            if let Some(kv) = parse_hpc_line(line) {
                pending = Some(stats_from(&kv));
                continue;
            }
            let l = line.trim_start().to_ascii_lowercase();
            if let Some(rest) = l.strip_prefix("create table") {
                let rest = rest
                    .trim_start()
                    .strip_prefix("if not exists")
                    .unwrap_or(rest)
                    .trim_start();
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
                    .collect();
                if let Some(s) = pending.take() {
                    out.insert(name.rsplit('.').next().unwrap_or(&name).to_string(), s);
                }
            } else if !l.is_empty() && !l.starts_with("--") {
                pending = None;
            }
        }
    }
    for f in stats_files {
        let text = std::fs::read_to_string(f).with_context(|| f.display().to_string())?;
        let table: toml::Table = text
            .parse()
            .with_context(|| format!("parsing {}", f.display()))?;
        for (name, v) in table {
            let s: DataStats = v
                .try_into()
                .with_context(|| format!("[{name}] in {}", f.display()))?;
            out.insert(name.to_ascii_lowercase(), s);
        }
    }
    Ok(out)
}

/// Runs the model over every query statement in `files`.
pub fn analyze(
    files: &[PathBuf],
    schema_files: &[PathBuf],
    stats_files: &[PathBuf],
    model: &BottleneckNet,
    device: &Device,
    opts: Options,
) -> Result<Report> {
    let mut ddl = String::new();
    let mut raw_sources: Vec<String> = Vec::new();
    for f in schema_files {
        let text =
            std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
        ddl.push_str(&text);
        ddl.push('\n');
        raw_sources.push(text);
    }
    struct Query {
        file: usize,
        line: usize,
        column: usize,
        text: String,
        batch: usize,
    }
    let mut queries = Vec::new();
    let mut names = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        let src = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
        for s in statements(&src) {
            let text = &src[s.range.clone()];
            match first_word(text).as_str() {
                "create" | "alter" | "drop" => {
                    ddl.push_str(text);
                    ddl.push_str(";\n");
                }
                "select" | "insert" | "update" | "delete" => {
                    // `-- hpc: batch=N` on the line right above the statement
                    let before = &src[..s.range.start];
                    let batch = before
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .and_then(parse_hpc_line)
                        .and_then(|kv| kv.get("batch").and_then(|v| v.parse().ok()))
                        .unwrap_or(0);
                    queries.push(Query {
                        file: fi,
                        line: s.line,
                        column: s.column,
                        text: text.to_string(),
                        batch,
                    })
                }
                _ => {}
            }
        }
        raw_sources.push(src);
        names.push(f.display().to_string());
    }
    let schema = SchemaInfo::from_ddl(&ddl);
    if schema.is_empty() {
        bail!(
            "no CREATE TABLE statements found; put the schema next to the queries or pass --schema <file.cql>"
        );
    }
    let stats = collect_stats(&raw_sources, stats_files)?;

    let mut report = Report {
        files: files.len(),
        statements: queries.len(),
        ..Report::default()
    };
    for chunk in queries.chunks(256) {
        let mut ids = Vec::new();
        let mut ctx = Vec::new();
        let mut source = Vec::new();
        for q in chunk {
            ids.push(encode(&q.text, &schema));
            let table = schema.table_of(&q.text);
            let known = table
                .and_then(|(n, _)| stats.get(n))
                .filter(|s| s.partitions.is_some() && s.rows_per_partition.is_some());
            let f = Facts {
                batch_n: q.batch,
                ..facts(&q.text)
            };
            match known {
                Some(s) => {
                    let mut s = s.clone();
                    if s.row_bytes.is_none() {
                        s.row_bytes = table.map(|(_, t)| t.row_bytes);
                    }
                    s.indexed |= table.is_some_and(|(_, t)| t.indexed);
                    ctx.push(context(Some(&s), &f));
                    source.push(format!(
                        "{} partitions x {} rows per partition",
                        fmt_count(s.partitions.unwrap_or(0.0)),
                        fmt_count(s.rows_per_partition.unwrap_or(0.0))
                    ));
                    report.with_data_stats += 1;
                }
                None => {
                    ctx.push(context(None, &f));
                    source.push("no data statistics (add `-- hpc: partitions=.. rows=..` above the CREATE TABLE or pass --stats)".to_string());
                }
            }
        }
        let seqs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        let (p, c) = model.predict(batch_tokens(&seqs, device), batch_ctx(&ctx, device));
        let p = p
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let c = c
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for (row, q) in chunk.iter().enumerate() {
            let cost = 10f32.powf(c[row]);
            let mut any = false;
            for (k, label) in LABELS.iter().enumerate() {
                let prob = p[row * LABELS.len() + k];
                if prob >= opts.threshold {
                    any = true;
                    report.findings.push(Finding {
                        file: names[q.file].clone(),
                        line: q.line,
                        column: q.column,
                        label,
                        probability: prob,
                        cost_ratio: cost,
                        statement: q.text.clone(),
                        data_source: source[row].clone(),
                    });
                }
            }
            if !any && cost >= opts.min_cost {
                report.slow_statements.push(Finding {
                    file: names[q.file].clone(),
                    line: q.line,
                    column: q.column,
                    label: "Slow",
                    probability: 0.0,
                    cost_ratio: cost,
                    statement: q.text.clone(),
                    data_source: source[row].clone(),
                });
            }
        }
    }
    report
        .findings
        .sort_by(|a, b| (&a.file, a.line, a.label).cmp(&(&b.file, b.line, b.label)));
    report
        .slow_statements
        .sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    Ok(report)
}

fn fmt_count(x: f64) -> String {
    if x >= 1e6 {
        format!("{:.1}M", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.0}k", x / 1e3)
    } else {
        format!("{x:.0}")
    }
}

/// `LargeResult` -> `large result`.
fn spaced(label: &str) -> String {
    let mut s = String::new();
    for (i, ch) in label.chars().enumerate() {
        if i > 0 && ch.is_uppercase() {
            s.push(' ');
        }
        s.extend(ch.to_lowercase());
    }
    s
}

fn snippet(f: &Finding, out: &mut String) {
    let first_line = f.statement.lines().next().unwrap_or("");
    let num = f.line.to_string();
    let pad = " ".repeat(num.len());
    out.push_str(&format!(
        "{pad}--> {}:{}:{}\n{pad} |\n{num} | {}\n{pad} |\n",
        f.file,
        f.line,
        f.column,
        first_line.trim_end()
    ));
}

/// Compiler-style diagnostics.
pub fn render(report: &Report) -> String {
    let mut out = String::new();
    for f in &report.findings {
        let help = LABELS
            .iter()
            .position(|l| *l == f.label)
            .map(|i| LABEL_HELP[i])
            .unwrap_or("");
        out.push_str(&format!(
            "warning: AI predicts a {} bottleneck ({:.0}% confidence, about {:.0}x the cost of a {}) on line {}\n",
            spaced(f.label),
            f.probability * 100.0,
            f.cost_ratio,
            if f.statement.trim_start().to_ascii_lowercase().starts_with("select") { "point read" } else { "plain insert" },
            f.line
        ));
        snippet(f, &mut out);
        let pad = " ".repeat(f.line.to_string().len());
        out.push_str(&format!(
            "{pad} = help: {help}\n{pad} = note: data shape: {}\n\n",
            f.data_source
        ));
    }
    for f in &report.slow_statements {
        out.push_str(&format!("note: predicted about {:.0}x the cost of a basic operation on line {} (no specific bottleneck class)\n", f.cost_ratio, f.line));
        snippet(f, &mut out);
        out.push('\n');
    }
    out.push_str(&format!(
        "cql-ai: checked {} statement(s) in {} file(s) ({} with data statistics), {} warning(s)\n",
        report.statements,
        report.files,
        report.with_data_stats,
        report.findings.len() + report.slow_statements.len()
    ));
    if report.with_data_stats == 0 && report.statements > 0 {
        out.push_str("hint: without data statistics the model only sees the statement; add `-- hpc: partitions=<n> rows=<n>` above a CREATE TABLE for sharper results\n");
    }
    out
}
