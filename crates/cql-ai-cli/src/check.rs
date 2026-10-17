//! Collect `.cql` files, run the model, render findings.

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use cql_ai_model::model::{AntiPatternNet, batch_tokens};
use cql_ai_model::tokenizer::{SchemaInfo, encode, statements};
use cql_ai_model::{LABEL_HELP, LABELS};
use serde::Serialize;
use std::path::PathBuf;
use walkdir::WalkDir;

/// One predicted anti-pattern.
#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub label: &'static str,
    pub probability: f32,
    pub statement: String,
}

#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub files: usize,
    pub statements: usize,
    pub findings: Vec<Finding>,
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

/// Runs the model over every query statement in `files`.
///
/// DDL found in the files (`CREATE ...`) plus `schema_files` provide the column roles.
pub fn analyze(
    files: &[PathBuf],
    schema_files: &[PathBuf],
    model: &AntiPatternNet,
    device: &Device,
    threshold: f32,
) -> Result<Report> {
    let mut ddl = String::new();
    for f in schema_files {
        ddl.push_str(
            &std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
        );
        ddl.push('\n');
    }
    struct Query {
        file: usize,
        line: usize,
        column: usize,
        text: String,
    }
    let mut queries = Vec::new();
    let mut sources = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        let src = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
        for s in statements(&src) {
            let text = &src[s.range.clone()];
            match first_word(text).as_str() {
                "create" => {
                    ddl.push_str(text);
                    ddl.push_str(";\n");
                }
                "select" | "insert" | "update" | "delete" | "begin" => queries.push(Query {
                    file: fi,
                    line: s.line,
                    column: s.column,
                    text: text.to_string(),
                }),
                _ => {}
            }
        }
        sources.push(f.display().to_string());
    }
    let schema = SchemaInfo::from_ddl(&ddl);
    if schema.is_empty() {
        bail!(
            "no CREATE TABLE statements found; put the schema next to the queries or pass --schema <file.cql>"
        );
    }

    let mut report = Report {
        files: files.len(),
        statements: queries.len(),
        findings: Vec::new(),
    };
    for chunk in queries.chunks(256) {
        let ids: Vec<Vec<u32>> = chunk.iter().map(|q| encode(&q.text, &schema)).collect();
        let seqs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        let probs = model
            .predict(batch_tokens(&seqs, device))
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for (row, q) in chunk.iter().enumerate() {
            for (c, label) in LABELS.iter().enumerate() {
                let p = probs[row * LABELS.len() + c];
                if p >= threshold {
                    report.findings.push(Finding {
                        file: sources[q.file].clone(),
                        line: q.line,
                        column: q.column,
                        label,
                        probability: p,
                        statement: q.text.clone(),
                    });
                }
            }
        }
    }
    report
        .findings
        .sort_by(|a, b| (&a.file, a.line, a.label).cmp(&(&b.file, b.line, b.label)));
    Ok(report)
}

/// `MissingPartitionKey` -> `Missing Partition Key`.
fn spaced(label: &str) -> String {
    let mut s = String::new();
    for (i, ch) in label.chars().enumerate() {
        if i > 0 && ch.is_uppercase() {
            s.push(' ');
        }
        s.push(ch);
    }
    s
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
        let first_line = f.statement.lines().next().unwrap_or("");
        let num = f.line.to_string();
        let pad = " ".repeat(num.len());
        out.push_str(&format!(
            "warning: AI detects {:.0}% probability of {} on line {}\n{pad}--> {}:{}:{}\n{pad} |\n{num} | {}\n{pad} |\n{pad} = help: {help}\n\n",
            f.probability * 100.0,
            spaced(f.label),
            f.line,
            f.file,
            f.line,
            f.column,
            first_line.trim_end(),
        ));
    }
    out.push_str(&format!(
        "cql-ai: checked {} statement(s) in {} file(s), {} warning(s)\n",
        report.statements,
        report.files,
        report.findings.len()
    ));
    out
}
