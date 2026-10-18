use burn::prelude::*;
use cql_ai_model::features::{DataStats, Facts, N_CTX, context};
use cql_ai_model::io::{load, save};
use cql_ai_model::model::{ModelConfig, batch_ctx, batch_tokens};
use cql_ai_model::tokenizer::{SchemaInfo, encode};
use cql_ai_model::{LABELS, N_LABELS};

fn tiny() -> ModelConfig {
    ModelConfig::new()
        .with_d_model(32)
        .with_d_ff(64)
        .with_n_layers(2)
}

fn ctx_of(rows: usize) -> Vec<[f32; N_CTX]> {
    let s = DataStats {
        partitions: Some(100.0),
        rows_per_partition: Some(1000.0),
        ..Default::default()
    };
    vec![context(Some(&s), &Facts::default()); rows]
}

fn run(
    model: &cql_ai_model::model::BottleneckNet,
    seqs: &[&[u32]],
    device: &Device,
) -> (Vec<f32>, Vec<f32>) {
    let (p, c) = model.predict(
        batch_tokens(seqs, device),
        batch_ctx(&ctx_of(seqs.len()), device),
    );
    (
        p.into_data().try_to_vec::<f32>().unwrap(),
        c.into_data().try_to_vec::<f32>().unwrap(),
    )
}

#[test]
fn forward_shapes_and_probability_range() {
    let device = Device::flex();
    let model = tiny().init(&device);
    let a = encode("SELECT a FROM t WHERE k = ?", &SchemaInfo::default());
    let b = encode("SELECT a FROM t", &SchemaInfo::default());
    let (p, c) = run(&model, &[&a, &b], &device);
    assert_eq!(p.len(), 2 * N_LABELS);
    assert_eq!(c.len(), 2);
    assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
    assert_eq!(LABELS.len(), N_LABELS);
}

#[test]
fn padding_does_not_change_the_result() {
    let device = Device::flex();
    let model = tiny().init(&device);
    let short = encode("SELECT a FROM t", &SchemaInfo::default());
    let long = encode(
        "SELECT a, b, c, d FROM t WHERE k = ? AND j = ?",
        &SchemaInfo::default(),
    );
    let (alone, ca) = run(&model, &[&short], &device);
    let (padded, cp) = run(&model, &[&short, &long], &device);
    for (x, y) in alone.iter().zip(&padded[..N_LABELS]) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
    assert!((ca[0] - cp[0]).abs() < 1e-4);
}

#[test]
fn the_data_context_changes_the_prediction() {
    let device = Device::flex();
    let model = tiny().init(&device);
    let ids = encode("SELECT * FROM t WHERE k = ?", &SchemaInfo::default());
    let with = model
        .predict(
            batch_tokens(&[&ids], &device),
            batch_ctx(&ctx_of(1), &device),
        )
        .1
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let blind = model
        .predict(
            batch_tokens(&[&ids], &device),
            batch_ctx(&[context(None, &Facts::default())], &device),
        )
        .1
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_ne!(with, blind);
}

#[test]
fn safetensors_roundtrip_is_exact() {
    let device = Device::flex();
    let cfg = tiny();
    let model = cfg.init(&device);
    let ids = encode(
        "UPDATE t SET tags = {'a'} WHERE k = 1",
        &SchemaInfo::default(),
    );
    let want = run(&model, &[&ids], &device);

    let dir = tempfile::tempdir().unwrap();
    save(&model, &cfg, dir.path()).unwrap();
    assert!(dir.path().join("model.safetensors").exists());
    let (loaded, loaded_cfg) = load(dir.path(), &device).unwrap();
    assert_eq!(loaded_cfg.d_model, 32);
    assert_eq!(want, run(&loaded, &[&ids], &device));
}
