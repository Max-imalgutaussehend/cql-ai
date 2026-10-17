//! Sanity checks against the *trained* model. Skipped unless `CQL_AI_MODEL_DIR` points at a
//! directory with `model.safetensors` + `config.json` (for example `artifacts/`).

use burn::prelude::*;
use cql_ai_cli::check;
use cql_ai_model::io;
use std::time::Instant;

const SCHEMA: &str = "
CREATE TABLE events (tenant text, day int, ts bigint, body text, tags set<text>, PRIMARY KEY ((tenant, day), ts)) WITH CLUSTERING ORDER BY (ts DESC);
CREATE TABLE users (user_id uuid PRIMARY KEY, name text, email text);
";

fn labels_for(report: &check::Report, line: usize) -> Vec<&'static str> {
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
        "SELECT body FROM events WHERE tenant = ? AND day = ? AND ts > ?;", // 1: fine
        "SELECT body FROM events WHERE tenant = ?;", // 2: partition key incomplete
        "SELECT body FROM events WHERE body = ? ALLOW FILTERING;", // 3: filtering on non-key
        "UPDATE events SET tags = {'a', 'b'} WHERE tenant = ? AND day = ? AND ts = ?;", // 4: overwrite collection
        "SELECT name FROM users WHERE user_id = ?;",                                    // 5: fine
    ];
    std::fs::write(tmp.path().join("schema.cql"), SCHEMA).unwrap();
    std::fs::write(tmp.path().join("q.cql"), queries.join("\n")).unwrap();

    let started = Instant::now();
    let files = check::collect_files(&[tmp.path().join("q.cql")]).unwrap();
    let report =
        check::analyze(&files, &[tmp.path().join("schema.cql")], &net, &device, 0.5).unwrap();
    eprintln!("analysis took {:?}", started.elapsed());

    assert!(
        labels_for(&report, 1).is_empty(),
        "{:?}",
        labels_for(&report, 1)
    );
    assert!(labels_for(&report, 2).contains(&"MissingPartitionKey"));
    assert!(labels_for(&report, 3).contains(&"InefficientAllowFiltering"));
    assert!(labels_for(&report, 4).contains(&"LargeCollectionUpdate"));
    assert!(
        labels_for(&report, 5).is_empty(),
        "{:?}",
        labels_for(&report, 5)
    );
}
