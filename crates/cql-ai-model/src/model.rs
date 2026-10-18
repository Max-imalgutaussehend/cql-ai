//! The Transformer encoder: lexer tokens + data-shape context -> bottleneck logits and a cost estimate.

use crate::LABELS;
use crate::features::N_CTX;
use crate::vocab::{PAD, VOCAB_SIZE};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

/// Hyper-parameters. The defaults give a ~0.6M parameter model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "VOCAB_SIZE")]
    pub vocab_size: usize,
    #[config(default = 96)]
    pub max_len: usize,
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 256)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = "LABELS.len()")]
    pub n_labels: usize,
    #[config(default = "N_CTX")]
    pub n_ctx: usize,
}

/// Token + position embeddings (+ data-shape context) -> Transformer encoder -> masked mean pool -> two heads.
#[derive(Module, Debug)]
pub struct BottleneckNet {
    tok: Embedding,
    pos: Embedding,
    ctx: Linear,
    encoder: TransformerEncoder,
    norm: LayerNorm,
    labels: Linear,
    cost: Linear,
}

/// Raw model outputs.
pub struct Output {
    /// Bottleneck logits `[batch, n_labels]` (sigmoid for probabilities).
    pub logits: Tensor<2>,
    /// `log10(cost / baseline)`, `[batch, 1]`: how many times slower than a point read (reads) or a plain insert (writes).
    pub cost: Tensor<2>,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> BottleneckNet {
        BottleneckNet {
            tok: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            pos: EmbeddingConfig::new(self.max_len, self.d_model).init(device),
            ctx: LinearConfig::new(self.n_ctx, self.d_model).init(device),
            encoder: TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            labels: LinearConfig::new(self.d_model, self.n_labels).init(device),
            cost: LinearConfig::new(self.d_model, 1).init(device),
        }
    }
}

impl BottleneckNet {
    /// `tokens`: `[batch, seq]` ids padded with [`PAD`]; `ctx`: `[batch, n_ctx]` from [`crate::features::context`].
    pub fn forward(&self, tokens: Tensor<2, Int>, ctx: Tensor<2>) -> Output {
        let [_, seq] = tokens.dims();
        let device = tokens.device();
        let pad = tokens.clone().equal_elem(i64::from(PAD));
        let positions = Tensor::<1, Int>::arange(0..seq as i64, &device).unsqueeze::<2>();

        // The context vector is added to every position, so attention can combine "what the query does" with
        // "how much data it touches".
        let c = self.ctx.forward(ctx).unsqueeze_dim::<3>(1);
        let x = self.tok.forward(tokens) + self.pos.forward(positions) + c;
        let x = self
            .encoder
            .forward(TransformerEncoderInput::new(x).mask_pad(pad.clone()));
        let x = self.norm.forward(x);

        let keep = pad.bool_not().float().unsqueeze_dim::<3>(2);
        let sum = (x * keep.clone()).sum_dim(1);
        let count = keep.sum_dim(1).clamp_min(1.0);
        let pooled = (sum / count).squeeze_dim::<2>(1);
        Output {
            logits: self.labels.forward(pooled.clone()),
            cost: self.cost.forward(pooled),
        }
    }

    /// Per-label probabilities and the cost estimate.
    pub fn predict(&self, tokens: Tensor<2, Int>, ctx: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let o = self.forward(tokens, ctx);
        (burn::tensor::activation::sigmoid(o.logits), o.cost)
    }
}

/// Pads/stacks encoded statements into a `[batch, longest]` id tensor.
pub fn batch_tokens(seqs: &[&[u32]], device: &Device) -> Tensor<2, Int> {
    let len = seqs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    let mut flat = Vec::with_capacity(seqs.len() * len);
    for s in seqs {
        flat.extend(s.iter().map(|&i| i64::from(i)));
        flat.extend(std::iter::repeat_n(i64::from(PAD), len - s.len()));
    }
    Tensor::<1, Int>::from_ints(flat.as_slice(), device).reshape([seqs.len(), len])
}

/// Stacks context vectors into a `[batch, N_CTX]` tensor.
pub fn batch_ctx(rows: &[[f32; N_CTX]], device: &Device) -> Tensor<2> {
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::<1>::from_floats(flat.as_slice(), device).reshape([rows.len(), N_CTX])
}
