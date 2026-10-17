use burn::prelude::*;
use cql_ai_model::io::{load, save};
use cql_ai_model::model::{ModelConfig, batch_tokens};
use cql_ai_model::tokenizer::{SchemaInfo, encode};

fn tiny() -> ModelConfig {
    ModelConfig::new()
        .with_d_model(32)
        .with_d_ff(64)
        .with_n_layers(2)
}

#[test]
fn forward_shape_and_probability_range() {
    let device = Device::flex();
    let model = tiny().init(&device);
    let a = encode("SELECT a FROM t WHERE k = ?", &SchemaInfo::default());
    let b = encode("SELECT a FROM t", &SchemaInfo::default());
    let probs = model
        .predict(batch_tokens(&[&a, &b], &device))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(probs.len(), 2 * 6);
    assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
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
    let alone = model
        .predict(batch_tokens(&[&short], &device))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let padded = model
        .predict(batch_tokens(&[&short, &long], &device))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    for (x, y) in alone.iter().zip(&padded[..6]) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
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
    let want = model
        .predict(batch_tokens(&[&ids], &device))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    save(&model, &cfg, dir.path()).unwrap();
    assert!(dir.path().join("model.safetensors").exists());
    let (loaded, loaded_cfg) = load(dir.path(), &device).unwrap();
    assert_eq!(loaded_cfg.d_model, 32);
    let got = loaded
        .predict(batch_tokens(&[&ids], &device))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(want, got);
}
