//! Training loop (AdamW, warm-up + cosine decay; BCE for the bottleneck labels, MSE for the cost) and evaluation.

use crate::dataset::Example;
use anyhow::Result;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::log_sigmoid;
use cql_ai_model::LABELS;
use cql_ai_model::features::N_CTX;
use cql_ai_model::model::{BottleneckNet, ModelConfig, batch_ctx, batch_tokens};
use rand::prelude::*;
use std::time::Instant;

pub const L: usize = LABELS.len();

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f32,
    pub seed: u64,
    /// Probability that a training example is shown without data statistics.
    pub blind_prob: f64,
    /// Weight of the cost regression in the loss.
    pub cost_weight: f32,
}

/// Evaluation result.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub n: usize,
    pub positives: [usize; L],
    pub precision: [f32; L],
    pub recall: [f32; L],
    pub f1: [f32; L],
    /// Mean F1 over labels that occur in the evaluated set.
    pub macro_f1: f32,
    /// Mean absolute error of `log10(cost / baseline)`.
    pub cost_mae: f32,
    /// Spearman rank correlation between predicted and measured cost.
    pub spearman: f32,
}

impl Metrics {
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (i, label) in LABELS.iter().enumerate() {
            s.push_str(&format!(
                "    {:<16} n+={:<5} precision {:.3}  recall {:.3}  f1 {:.3}\n",
                label, self.positives[i], self.precision[i], self.recall[i], self.f1[i]
            ));
        }
        s.push_str(&format!(
            "    macro-F1 {:.3}   cost MAE {:.3} (log10)   Spearman {:.3}   [{} statements]\n",
            self.macro_f1, self.cost_mae, self.spearman, self.n
        ));
        s
    }
}

/// Fraction-based scoring of predicted label sets and costs against the examples.
pub fn score(examples: &[&Example], pred_labels: &[[bool; L]], pred_cost: &[f32]) -> Metrics {
    let (mut tp, mut fp, mut fneg, mut pos) = ([0f32; L], [0f32; L], [0f32; L], [0usize; L]);
    for (e, p) in examples.iter().zip(pred_labels) {
        for c in 0..L {
            pos[c] += usize::from(e.labels[c]);
            match (p[c], e.labels[c]) {
                (true, true) => tp[c] += 1.0,
                (true, false) => fp[c] += 1.0,
                (false, true) => fneg[c] += 1.0,
                _ => {}
            }
        }
    }
    let mut m = Metrics {
        n: examples.len(),
        positives: pos,
        precision: [0.; L],
        recall: [0.; L],
        f1: [0.; L],
        macro_f1: 0.,
        cost_mae: 0.,
        spearman: 0.,
    };
    let mut present = 0;
    for c in 0..L {
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
        if pos[c] > 0 {
            m.macro_f1 += m.f1[c];
            present += 1;
        }
    }
    m.macro_f1 /= present.max(1) as f32;
    if !pred_cost.is_empty() {
        m.cost_mae = examples
            .iter()
            .zip(pred_cost)
            .map(|(e, p)| (e.cost - p).abs())
            .sum::<f32>()
            / examples.len() as f32;
        let truth: Vec<f32> = examples.iter().map(|e| e.cost).collect();
        m.spearman = spearman(&truth, pred_cost);
    }
    m
}

fn ranks(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0f32; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f32 / 2.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

pub fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f32;
    let (ma, mb) = (ra.iter().sum::<f32>() / n, rb.iter().sum::<f32>() / n);
    let (mut num, mut da, mut db) = (0f32, 0f32, 0f32);
    for (x, y) in ra.iter().zip(&rb) {
        num += (x - ma) * (y - mb);
        da += (x - ma).powi(2);
        db += (y - mb).powi(2);
    }
    if da > 0. && db > 0. {
        num / (da * db).sqrt()
    } else {
        0.0
    }
}

fn tokens(ex: &[&Example], device: &Device) -> Tensor<2, Int> {
    let seqs: Vec<&[u32]> = ex.iter().map(|e| e.ids.as_slice()).collect();
    batch_tokens(&seqs, device)
}

/// Probabilities (thresholded at `threshold`) and cost predictions for `examples`.
pub fn predict(
    model: &BottleneckNet,
    examples: &[&Example],
    blind: bool,
    threshold: f32,
    device: &Device,
) -> (Vec<[bool; L]>, Vec<f32>, Vec<[f32; L]>) {
    let (mut labels, mut cost, mut probs) = (Vec::new(), Vec::new(), Vec::new());
    for chunk in examples.chunks(256) {
        let ctx: Vec<[f32; N_CTX]> = chunk
            .iter()
            .map(|e| if blind { e.ctx_blind } else { e.ctx })
            .collect();
        let (p, c) = model.predict(tokens(chunk, device), batch_ctx(&ctx, device));
        let p = p.into_data().try_to_vec::<f32>().expect("f32");
        let c = c.into_data().try_to_vec::<f32>().expect("f32");
        for i in 0..chunk.len() {
            let mut row = [0f32; L];
            row.copy_from_slice(&p[i * L..(i + 1) * L]);
            labels.push(row.map(|x| x >= threshold));
            probs.push(row);
            cost.push(c[i]);
        }
    }
    (labels, cost, probs)
}

pub fn evaluate(
    model: &BottleneckNet,
    examples: &[&Example],
    blind: bool,
    device: &Device,
) -> Metrics {
    let (l, c, _) = predict(model, examples, blind, 0.5, device);
    score(examples, &l, &c)
}

/// Trains a fresh model; returns it on the (autodiff-free) `device`.
pub fn train(
    cfg: &ModelConfig,
    opts: &TrainOptions,
    train: &[&Example],
    val: &[&Example],
    device: &Device,
) -> Result<BottleneckNet> {
    let ad = device.clone().autodiff();
    let mut model = cfg.init(&ad);
    println!(
        "model: {} parameters, {} training statements",
        model.num_params(),
        train.len()
    );
    let mut optim = AdamWConfig::new()
        .with_weight_decay(opts.weight_decay)
        .init();
    let mut rng = StdRng::seed_from_u64(opts.seed);

    // positive weights against label imbalance, capped
    let mut pos = [0f32; L];
    for e in train {
        for c in 0..L {
            pos[c] += f32::from(e.labels[c]);
        }
    }
    let pos_w: Vec<f32> = pos
        .iter()
        .map(|&p| {
            if p > 0.0 {
                ((train.len() as f32 - p) / p).clamp(1.0, 12.0)
            } else {
                1.0
            }
        })
        .collect();

    let steps_per_epoch = train.len().div_ceil(opts.batch_size);
    let total = (steps_per_epoch * opts.epochs) as f64;
    let warmup = (total * 0.05).max(1.0);
    let mut step = 0f64;
    let started = Instant::now();
    let mut order: Vec<usize> = (0..train.len()).collect();
    for epoch in 1..=opts.epochs {
        order.shuffle(&mut rng);
        let mut loss_sum = 0f32;
        for chunk in order.chunks(opts.batch_size) {
            let ex: Vec<&Example> = chunk.iter().map(|&i| train[i]).collect();
            let lr = if step < warmup {
                opts.lr * (step + 1.0) / warmup
            } else {
                let p = (step - warmup) / (total - warmup).max(1.0);
                opts.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
            };
            let ctx: Vec<[f32; N_CTX]> = ex
                .iter()
                .map(|e| {
                    if rng.random_bool(opts.blind_prob) {
                        e.ctx_blind
                    } else {
                        e.ctx
                    }
                })
                .collect();
            let y: Vec<f32> = ex
                .iter()
                .flat_map(|e| e.labels.iter().map(|&b| f32::from(b)))
                .collect();
            let yt = Tensor::<1>::from_floats(y.as_slice(), &ad).reshape([ex.len(), L]);
            let target_cost = Tensor::<1>::from_floats(
                ex.iter().map(|e| e.cost).collect::<Vec<_>>().as_slice(),
                &ad,
            )
            .reshape([ex.len(), 1]);
            let w = Tensor::<1>::from_floats(pos_w.as_slice(), &ad).reshape([1, L]);

            let out = model.forward(tokens(&ex, &ad), batch_ctx(&ctx, &ad));
            let bce = -(yt.clone() * w * log_sigmoid(out.logits.clone())
                + (yt.neg() + 1.0) * log_sigmoid(out.logits.neg()))
            .mean();
            let mse = (out.cost - target_cost).powi_scalar(2).mean();
            let loss = bce + mse * opts.cost_weight;
            loss_sum += loss.clone().into_scalar::<f32>();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
            step += 1.0;
        }
        let m = evaluate(&model.valid(), val, false, device);
        println!(
            "epoch {epoch:>3}/{}  loss {:.4}  val macro-F1 {:.3}  cost MAE {:.3}  Spearman {:.3}  ({:.0}s)",
            opts.epochs,
            loss_sum / steps_per_epoch as f32,
            m.macro_f1,
            m.cost_mae,
            m.spearman,
            started.elapsed().as_secs_f32()
        );
    }
    Ok(model.valid())
}
