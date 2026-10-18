//! Sanity checks against the *trained* model. Skipped unless `CQL_AI_MODEL_DIR` points at a directory with
//! `model.safetensors` + `config.json` (for example `artifacts/`).

use burn::prelude::*;
use cql_ai_cli::check::{Options, analyze, collect_files};
use cql_ai_model::io;
use std::time::Instant;

const SCHEMA: &str = "
-- hpc: partitions=200 rows=10000 row_bytes=140
CREATE TABLE events (part text, clust text, data0 text, data1 text, data2 text, data3 text, PRIMARY KEY ((part), clust));
-- hpc: partitions=500000 rows=1 row_bytes=200
CREATE TABLE users (id text PRIMARY KEY, name text, email text);
";

fn labels_for(report: &cql_ai_cli::check::Report, line: usize) -> Vec<&'static str> {
    report
        .findings
        .iter()
        .filter(|f| f.line == line)
        .map(|f| f.label)
        .collect()
}

#[test]
fn trained_model_flags_the_obvious_cases() {
    let Some(dir) = std::env::var_os("CQL_AI_MODEL_DIR") else {
        eprintln!("skipped: CQL_AI_MODEL_DIR not set");
        return;
    };
    let device = Device::flex();
    let (net, _) = io::load(dir.as_ref(), &device).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let queries = [
        "SELECT * FROM users WHERE id = ?;",    // 1: point read, fine
        "SELECT * FROM events WHERE part = ?;", // 2: whole 10,000-row partition
        "SELECT * FROM events WHERE part = ? LIMIT 10;", // 3: fine
        "SELECT count(*) FROM events WHERE part = ?;", // 4: scans the partition
        "SELECT * FROM users LIMIT 100;",       // 5: no partition key
        "INSERT INTO users (id, name, email) VALUES (?, ?, ?) IF NOT EXISTS;", // 6: LWT
    ];
    std::fs::write(tmp.path().join("schema.cql"), SCHEMA).unwrap();
    std::fs::write(tmp.path().join("q.cql"), queries.join("\n")).unwrap();

    let started = Instant::now();
    let files = collect_files(&[tmp.path().join("q.cql")]).unwrap();
    let report = analyze(
        &files,
        &[tmp.path().join("schema.cql")],
        &[],
        &net,
        &device,
        Options {
            threshold: 0.5,
            min_cost: f32::MAX,
        },
    )
    .unwrap();
    eprintln!("analysis took {:?}", started.elapsed());
    for line in 1..=queries.len() {
        eprintln!("line {line}: {:?}", labels_for(&report, line));
    }
    assert!(
        labels_for(&report, 1).is_empty(),
        "point read: {:?}",
        labels_for(&report, 1)
    );
    assert!(
        !labels_for(&report, 2).is_empty(),
        "full 10,000-row partition read must be flagged"
    );
    assert!(
        labels_for(&report, 3).is_empty(),
        "LIMIT 10: {:?}",
        labels_for(&report, 3)
    );
    assert!(labels_for(&report, 4).contains(&"ServerScan"));
    // `SELECT * ... LIMIT 100` without a partition key measured only about 2x a point read: not a bottleneck by our thresholds.
    assert!(!labels_for(&report, 5).contains(&"LargeResult"));
    assert!(labels_for(&report, 6).contains(&"LightweightTxn"));
}
