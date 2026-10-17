use burn::prelude::*;
use cql_ai_cli::{cache, check};
use cql_ai_model::io;
use cql_ai_model::model::ModelConfig;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

const SCHEMA: &str = "CREATE TABLE events (tenant text, day int, ts bigint, body text, PRIMARY KEY ((tenant, day), ts));";
const QUERIES: &str = "-- name: A :many\nSELECT * FROM events WHERE tenant = ?;\n\nSELECT body FROM events WHERE tenant = ? AND day = ?;\n";

fn tiny_model(dir: &Path) {
    let cfg = ModelConfig::new()
        .with_d_model(32)
        .with_d_ff(64)
        .with_n_layers(1);
    let device = Device::flex();
    io::save(&cfg.init(&device), &cfg, dir).unwrap();
}

#[test]
fn analyze_reports_locations_and_renders_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    tiny_model(dir.path());
    let (net, _) = io::load(dir.path(), &Device::flex()).unwrap();
    std::fs::write(dir.path().join("schema.cql"), SCHEMA).unwrap();
    std::fs::write(dir.path().join("q.cql"), QUERIES).unwrap();

    let files = check::collect_files(&[dir.path().join("q.cql")]).unwrap();
    // Threshold 0 reports every label, which exercises the whole pipeline independent of weights.
    let report = check::analyze(
        &files,
        &[dir.path().join("schema.cql")],
        &net,
        &Device::flex(),
        0.0,
    )
    .unwrap();
    assert_eq!(report.statements, 2);
    assert_eq!(report.findings.len(), 2 * 6);
    assert!(
        report.findings.iter().any(|f| f.line == 2) && report.findings.iter().any(|f| f.line == 4)
    );

    let text = check::render(&report);
    assert!(text.contains("warning: AI detects "), "{text}");
    assert!(
        text.contains("% probability of Missing Partition Key on line 2"),
        "{text}"
    );
    assert!(
        text.contains("SELECT * FROM events WHERE tenant = ?"),
        "{text}"
    );
}

#[test]
fn schema_inside_the_query_file_is_used_and_missing_schema_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    tiny_model(dir.path());
    let (net, _) = io::load(dir.path(), &Device::flex()).unwrap();
    let both = dir.path().join("both.cql");
    std::fs::write(&both, format!("{SCHEMA}\n{QUERIES}")).unwrap();
    let r = check::analyze(std::slice::from_ref(&both), &[], &net, &Device::flex(), 1.1).unwrap();
    assert_eq!(r.statements, 2);

    let only_q = dir.path().join("only.cql");
    std::fs::write(&only_q, QUERIES).unwrap();
    let err = check::analyze(&[only_q], &[], &net, &Device::flex(), 0.5).unwrap_err();
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
    // SAFETY: single-threaded access to these variables within this test binary (only this test sets them).
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
            .all(|p| p.contains("Max-imalgutaussehend/cql-anti-pattern-transformer/resolve/main/")),
        "{served:?}"
    );

    // Second call must not need the (now closed) server.
    assert_eq!(cache::resolve_model(None, false, false).unwrap(), dir);
    assert!(
        cache::resolve_model(None, true, true).is_err(),
        "--offline --refresh cannot succeed"
    );
}
