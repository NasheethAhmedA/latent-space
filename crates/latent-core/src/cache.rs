//! The KV cache as a value. Cloning is cheap (tensors are shared until one
//! side appends), so a prefix read once can be continued many ways.

use crate::{Result, Tensor};

#[derive(Clone, Debug, Default)]
pub struct Cache {
    /// per layer (keys, values): [kv_heads, tokens, head_dim]
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// tokens fully committed (every layer has them)
    pub len: usize,
}

impl Cache {
    pub fn new(layers: usize) -> Self {
        Cache {
            layers: vec![None; layers],
            len: 0,
        }
    }

    /// Append this step's keys and values to layer `i`; returns everything so far.
    pub fn append(&mut self, i: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let kv = match self.layers[i].take() {
            None => (k, v),
            Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 1)?, Tensor::cat(&[&pv, &v], 1)?),
        };
        self.layers[i] = Some(kv.clone());
        Ok(kv)
    }

    /// Tokens held by layer `i` (it can run ahead of `len` mid-step).
    pub fn layer_len(&self, i: usize) -> usize {
        self.layers[i]
            .as_ref()
            .map(|(k, _)| k.dims()[1])
            .unwrap_or(0)
    }

    /// Keep only the first `n` tokens (rewind).
    pub fn truncate(&mut self, n: usize) -> Result<()> {
        for l in self.layers.iter_mut() {
            if let Some((k, v)) = l {
                if k.dims()[1] > n {
                    *l = if n == 0 {
                        None
                    } else {
                        Some((k.narrow(1, 0, n)?, v.narrow(1, 0, n)?))
                    };
                }
            }
        }
        self.len = self.len.min(n);
        Ok(())
    }
}
