use burn::prelude::*;
use cql_ai_cli::cache;
use cql_ai_cli::check::{Options, analyze, collect_files, render};
use cql_ai_model::io;
use cql_ai_model::model::ModelConfig;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

const SCHEMA: &str = "-- hpc: partitions=1000 rows=10000 row_bytes=300\nCREATE TABLE events (tenant text, day int, ts bigint, body text, PRIMARY KEY ((tenant, day), ts));";
const QUERIES: &str = "-- name: A :many\nSELECT * FROM events WHERE tenant = ? AND day = ?;\n\nSELECT body FROM events WHERE tenant = ? AND day = ? LIMIT 10;\n";

fn tiny_model(dir: &Path) {
    let cfg = ModelConfig::new()
        .with_d_model(32)
        .with_d_ff(64)
        .with_n_layers(1);
    let device = Device::flex();
    io::save(&cfg.init(&device), &cfg, dir).unwrap();
}

fn opts(threshold: f32) -> Options {
    Options {
        threshold,
        min_cost: f32::MAX,
    }
}

#[test]
fn analyze_reports_locations_data_source_and_renders_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    tiny_model(dir.path());
    let (net, _) = io::load(dir.path(), &Device::flex()).unwrap();
    std::fs::write(dir.path().join("schema.cql"), SCHEMA).unwrap();
    std::fs::write(dir.path().join("q.cql"), QUERIES).unwrap();

    let files = collect_files(&[dir.path().join("q.cql")]).unwrap();
    // Threshold 0 reports every label, which exercises the whole pipeline independent of weights.
    let report = analyze(
        &files,
        &[dir.path().join("schema.cql")],
        &[],
        &net,
        &Device::flex(),
        opts(0.0),
    )
    .unwrap();
    assert_eq!(report.statements, 2);
    assert_eq!(
        report.with_data_stats, 2,
        "the `-- hpc:` annotation above the CREATE TABLE is picked up"
    );
    assert_eq!(report.findings.len(), 2 * 6);
    assert!(
        report.findings.iter().any(|f| f.line == 2) && report.findings.iter().any(|f| f.line == 4)
    );

    let text = render(&report);
    assert!(text.contains("warning: AI predicts a "), "{text}");
    assert!(text.contains("bottleneck ("), "{text}");
    assert!(
        text.contains("1k partitions x 10k rows per partition"),
        "{text}"
    );
    assert!(
        text.contains("SELECT * FROM events WHERE tenant = ? AND day = ?"),
        "{text}"
    );
}

#[test]
fn stats_file_and_missing_stats_are_handled() {
    let dir = tempfile::tempdir().unwrap();
    tiny_model(dir.path());
    let (net, _) = io::load(dir.path(), &Device::flex()).unwrap();
    let both = dir.path().join("both.cql");
    std::fs::write(&both, format!("CREATE TABLE events (tenant text, day int, ts bigint, body text, PRIMARY KEY ((tenant, day), ts));\n{QUERIES}")).unwrap();
    let r = analyze(
        std::slice::from_ref(&both),
        &[],
        &[],
        &net,
        &Device::flex(),
        opts(1.1),
    )
    .unwrap();
    assert_eq!(r.statements, 2);
    assert_eq!(r.with_data_stats, 0);
    assert!(render(&r).contains("without data statistics"));

    let stats = dir.path().join("stats.toml");
    std::fs::write(
        &stats,
        "[events]\npartitions = 500.0\nrows_per_partition = 20000.0\n",
    )
    .unwrap();
    let r = analyze(&[both], &[], &[stats], &net, &Device::flex(), opts(1.1)).unwrap();
    assert_eq!(r.with_data_stats, 2);
}

#[test]
fn missing_schema_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    tiny_model(dir.path());
    let (net, _) = io::load(dir.path(), &Device::flex()).unwrap();
    let only_q = dir.path().join("only.cql");
    std::fs::write(&only_q, QUERIES).unwrap();
    let err = analyze(&[only_q], &[], &[], &net, &Device::flex(), opts(0.5)).unwrap_err();
    assert!(err.to_string().contains("no CREATE TABLE"), "{err}");
}

#[test]
fn model_is_downloaded_once_and_then_served_from_cache() {
    let src = tempfile::tempdir().unwrap();
    tiny_model(src.path());
    let weights = std::fs::read(src.path().join("model.safetensors")).unwrap();
    let config = std::fs::read(src.path().join("config.json")).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut served = Vec::new();
        for stream in listener.incoming().take(2) {
            let mut s = stream.unwrap();
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = req.split_whitespace().nth(1).unwrap().to_string();
            let body = if path.ends_with("model.safetensors") {
                &weights
            } else {
                &config
            };
            write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            s.write_all(body).unwrap();
            served.push(path);
        }
        served
    });

    let cache_dir = tempfile::tempdir().unwrap();
    // SAFETY: only this test sets these variables, and it does so before any thread of its own reads them.
    unsafe {
        std::env::set_var("CQL_AI_CACHE", cache_dir.path());
        std::env::set_var("CQL_AI_HF_BASE", format!("http://{addr}"));
    }
    let dir = cache::resolve_model(None, false, false).unwrap();
    assert!(dir.starts_with(cache_dir.path()));
    io::load(&dir, &Device::flex()).unwrap();
    let served = server.join().unwrap();
    assert!(
        served
            .iter()
            .all(|p| p.contains("Max-imumlikelihood/cql-hpc-bottleneck-transformer/resolve/main/")),
        "{served:?}"
    );
    assert_eq!(cache::resolve_model(None, false, false).unwrap(), dir);
    assert!(cache::resolve_model(None, true, true).is_err());
}
