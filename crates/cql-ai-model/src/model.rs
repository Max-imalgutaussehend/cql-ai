//! The Transformer encoder.

use crate::LABELS;
use crate::vocab::{PAD, VOCAB_SIZE};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

/// Hyper-parameters. The defaults give a ~0.9M parameter model.
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
}

/// Token + position embeddings -> Transformer encoder -> masked mean pool -> multi-label head.
#[derive(Module, Debug)]
pub struct AntiPatternNet {
    tok: Embedding,
    pos: Embedding,
    encoder: TransformerEncoder,
    norm: LayerNorm,
    head: Linear,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> AntiPatternNet {
        AntiPatternNet {
            tok: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            pos: EmbeddingConfig::new(self.max_len, self.d_model).init(device),
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
            head: LinearConfig::new(self.d_model, self.n_labels).init(device),
        }
    }
}

impl AntiPatternNet {
    /// `tokens`: `[batch, seq]` ids padded with [`PAD`]. Returns logits `[batch, n_labels]`
    /// (apply a sigmoid for per-label probabilities).
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<2> {
        let [_, seq] = tokens.dims();
        let device = tokens.device();
        let pad = tokens.clone().equal_elem(i64::from(PAD));
        let positions = Tensor::<1, Int>::arange(0..seq as i64, &device).unsqueeze::<2>();

        let x = self.tok.forward(tokens) + self.pos.forward(positions);
        let x = self
            .encoder
            .forward(TransformerEncoderInput::new(x).mask_pad(pad.clone()));
        let x = self.norm.forward(x);

        // Masked mean over the sequence.
        let keep = pad.bool_not().float().unsqueeze_dim::<3>(2);
        let sum = (x * keep.clone()).sum_dim(1);
        let count = keep.sum_dim(1).clamp_min(1.0);
        let pooled = (sum / count).squeeze_dim::<2>(1);
        self.head.forward(pooled)
    }

    /// Per-label probabilities for a batch of padded sequences.
    pub fn predict(&self, tokens: Tensor<2, Int>) -> Tensor<2> {
        burn::tensor::activation::sigmoid(self.forward(tokens))
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
