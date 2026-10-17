//! Training loop (AdamW, linear warm-up + cosine decay, BCE-with-logits) and evaluation.

use crate::synth::{Dataset, N_LABELS};
use anyhow::Result;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use cql_ai_model::LABELS;
use cql_ai_model::model::{AntiPatternNet, ModelConfig, batch_tokens};
use rand::prelude::*;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f32,
    pub seed: u64,
}

/// Per-label and macro metrics at threshold 0.5.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub precision: [f32; N_LABELS],
    pub recall: [f32; N_LABELS],
    pub f1: [f32; N_LABELS],
    pub macro_f1: f32,
    /// Fraction of samples whose whole label set is predicted exactly.
    pub exact_match: f32,
}

impl Metrics {
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (i, label) in LABELS.iter().enumerate() {
            s.push_str(&format!(
                "  {:<28} precision {:.3}  recall {:.3}  f1 {:.3}\n",
                label, self.precision[i], self.recall[i], self.f1[i]
            ));
        }
        s.push_str(&format!(
            "  macro-F1 {:.4}   exact-match {:.4}\n",
            self.macro_f1, self.exact_match
        ));
        s
    }
}

fn targets(ds: &Dataset, idx: &[usize], device: &Device) -> Tensor<2, Int> {
    let flat: Vec<i64> = idx
        .iter()
        .flat_map(|&i| ds.labels[i].iter().map(|&b| i64::from(b)))
        .collect();
    Tensor::<1, Int>::from_ints(flat.as_slice(), device).reshape([idx.len(), N_LABELS])
}

fn tokens(ds: &Dataset, idx: &[usize], device: &Device) -> Tensor<2, Int> {
    let seqs: Vec<&[u32]> = idx.iter().map(|&i| ds.ids[i].as_slice()).collect();
    batch_tokens(&seqs, device)
}

/// Evaluates `model` (must live on a non-autodiff device, e.g. via `valid()`).
pub fn evaluate(model: &AntiPatternNet, ds: &Dataset, device: &Device, batch: usize) -> Metrics {
    let (mut tp, mut fp, mut fneg) = ([0f32; N_LABELS], [0f32; N_LABELS], [0f32; N_LABELS]);
    let mut exact = 0usize;
    let order: Vec<usize> = (0..ds.len()).collect();
    for chunk in order.chunks(batch) {
        let probs = model
            .predict(tokens(ds, chunk, device))
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 probabilities");
        for (row, &i) in chunk.iter().enumerate() {
            let mut all = true;
            for c in 0..N_LABELS {
                let pred = probs[row * N_LABELS + c] >= 0.5;
                let truth = ds.labels[i][c];
                match (pred, truth) {
                    (true, true) => tp[c] += 1.0,
                    (true, false) => fp[c] += 1.0,
                    (false, true) => fneg[c] += 1.0,
                    _ => {}
                }
                all &= pred == truth;
            }
            exact += usize::from(all);
        }
    }
    let mut m = Metrics {
        precision: [0.; N_LABELS],
        recall: [0.; N_LABELS],
        f1: [0.; N_LABELS],
        macro_f1: 0.,
        exact_match: 0.,
    };
    for c in 0..N_LABELS {
        m.precision[c] = if tp[c] + fp[c] > 0. {
            tp[c] / (tp[c] + fp[c])
        } else {
            0.
        };
        m.recall[c] = if tp[c] + fneg[c] > 0. {
            tp[c] / (tp[c] + fneg[c])
        } else {
            0.
        };
        let d = m.precision[c] + m.recall[c];
        m.f1[c] = if d > 0. {
            2. * m.precision[c] * m.recall[c] / d
        } else {
            0.
        };
    }
    m.macro_f1 = m.f1.iter().sum::<f32>() / N_LABELS as f32;
    m.exact_match = exact as f32 / ds.len().max(1) as f32;
    m
}

/// Trains a fresh model; returns it on the (autodiff-free) `device`.
pub fn train(
    cfg: &ModelConfig,
    opts: &TrainOptions,
    train: &Dataset,
    val: &Dataset,
    device: &Device,
) -> Result<AntiPatternNet> {
    let ad = device.clone().autodiff();
    let mut model = cfg.init(&ad);
    println!("model: {} parameters", model.num_params());
    let mut optim = AdamWConfig::new()
        .with_weight_decay(opts.weight_decay)
        .init();
    let loss_fn = burn::nn::loss::BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&ad);

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let steps_per_epoch = train.len().div_ceil(opts.batch_size);
    let total = (steps_per_epoch * opts.epochs) as f64;
    let warmup = (total * 0.05).max(1.0);
    let mut step = 0f64;
    let start = Instant::now();

    for epoch in 1..=opts.epochs {
        let mut order: Vec<usize> = (0..train.len()).collect();
        order.shuffle(&mut rng);
        let mut loss_sum = 0f32;
        for chunk in order.chunks(opts.batch_size) {
            let lr = if step < warmup {
                opts.lr * (step + 1.0) / warmup
            } else {
                let p = (step - warmup) / (total - warmup).max(1.0);
                opts.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
            };
            let logits = model.forward(tokens(train, chunk, &ad));
            let loss = loss_fn.forward(logits, targets(train, chunk, &ad));
            loss_sum += loss.clone().into_scalar::<f32>();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
            step += 1.0;
        }
        let m = evaluate(&model.valid(), val, &device.clone(), 512);
        println!(
            "epoch {epoch}/{}  train-loss {:.4}  val macro-F1 {:.4}  exact {:.4}  ({:.0}s)",
            opts.epochs,
            loss_sum / steps_per_epoch as f32,
            m.macro_f1,
            m.exact_match,
            start.elapsed().as_secs_f32()
        );
    }
    Ok(model.valid())
}
