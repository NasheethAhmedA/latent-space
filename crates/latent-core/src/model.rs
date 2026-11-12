//! A model is its stages. Each is a plain function of tensors, so a [`Run`]
//! can stop between any two and let the program look or change things.
//!
//! [`Run`]: crate::Run

use crate::{Cache, Result, Tensor};
use tokenizers::Tokenizer;

/// Shape facts every algorithm may need.
#[derive(Clone, Debug)]
pub struct Spec {
    /// a stable name, e.g. "gemma-3-270m-it" (vectors are tagged with it)
    pub id: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// special tokens
    pub bos: u32,
    pub eos: Vec<u32>,
}

pub trait Model: Send + Sync {
    fn spec(&self) -> &Spec;
    fn tokenizer(&self) -> &Tokenizer;

    /// Token ids → the stream the first layer reads: [n, hidden].
    fn embed(&self, ids: &[u32]) -> Result<Tensor>;

    /// Attention of layer `i` over the stream `x` ([n, hidden]) at positions
    /// `pos0..pos0+n`: appends this step's keys and values to `cache` and
    /// returns (probabilities [heads, n, total], values [kv_heads, total, head_dim]).
    fn attention(
        &self,
        i: usize,
        x: &Tensor,
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<(Tensor, Tensor)>;

    /// Mix values by the probabilities into the layer's contribution to the
    /// stream ([n, hidden]); probabilities may have been edited.
    fn attention_out(&self, i: usize, probs: &Tensor, values: &Tensor) -> Result<Tensor>;

    /// The MLP of layer `i` on the stream after attention: its contribution [n, hidden].
    fn mlp(&self, i: usize, mid: &Tensor) -> Result<Tensor>;

    fn final_norm(&self, x: &Tensor) -> Result<Tensor>;

    /// The normed stream → logits over the vocabulary: [n, vocab].
    fn unembed(&self, x: &Tensor) -> Result<Tensor>;

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer()
            .encode(text, false)
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec())
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer()
            .decode(ids, true)
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))
    }
    /// One token's text (spaces and all), for showing per-token results.
    fn piece(&self, id: u32) -> String {
        self.tokenizer()
            .id_to_token(id)
            .unwrap_or_default()
            .replace('▁', " ")
    }
}
