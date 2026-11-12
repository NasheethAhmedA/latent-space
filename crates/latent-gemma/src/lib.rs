//! Gemma 3 (text) as stages for `latent-core`: embeddings (scaled by
//! √hidden), and per layer RMSNorm (1 + w) → attention with q/k norms,
//! rotary positions (10k local / 1M global), sliding window on local layers,
//! grouped KV heads → post-attention norm; then pre-FF norm → gated GELU MLP →
//! post-FF norm. Final norm, tied unembedding. Runs on CPU in f32 (the weights
//! are bf16 on disk).
//!
//! The attention probabilities are materialised (not fused), so a program
//! can read and edit them like any other site.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use latent_core::{err, Cache, Error, Model, Result, Spec};
use std::path::Path;
use tokenizers::Tokenizer;

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    in_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ff_norm: Tensor,
    post_ff_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    sliding: bool,
}

pub struct Gemma3 {
    spec: Spec,
    tok: Tokenizer,
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    eps: f64,
    window: usize,
    rope_local: f64,
    rope_global: f64,
    scale: f64,
    dev: Device,
}

fn rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    // Gemma's RMSNorm: x / rms(x) * (1 + w)
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let y = x.broadcast_div(&(ms + eps)?.sqrt()?)?;
    Ok(y.broadcast_mul(&(w + 1.0)?)?)
}

impl Gemma3 {
    /// Load from a folder with config.json, tokenizer.json, model.safetensors.
    pub fn load(dir: &Path) -> Result<Gemma3> {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| Error::Msg(format!("{}: {e}", dir.display())))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))?;
        let u = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| Error::Msg(format!("config.json: no {k}")))
        };
        let f = |k: &str, d: f64| cfg[k].as_f64().unwrap_or(d);
        let dev = Device::Cpu;
        let files = [dir.join("model.safetensors")];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &dev)? };
        let (hidden, heads, kv_heads, head_dim, inter, layers, vocab) = (
            u("hidden_size")?,
            u("num_attention_heads")?,
            u("num_key_value_heads")?,
            u("head_dim")?,
            u("intermediate_size")?,
            u("num_hidden_layers")?,
            u("vocab_size")?,
        );
        let types: Vec<String> = cfg["layer_types"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let m = vb.pp("model");
        let lin = |vb: VarBuilder, o: usize, i: usize| -> Result<Linear> {
            Ok(Linear::new(vb.get((o, i), "weight")?, None))
        };
        let mut ls = vec![];
        for i in 0..layers {
            let l = m.pp(format!("layers.{i}"));
            let a = l.pp("self_attn");
            ls.push(Layer {
                q: lin(a.pp("q_proj"), heads * head_dim, hidden)?,
                k: lin(a.pp("k_proj"), kv_heads * head_dim, hidden)?,
                v: lin(a.pp("v_proj"), kv_heads * head_dim, hidden)?,
                o: lin(a.pp("o_proj"), hidden, heads * head_dim)?,
                q_norm: a.pp("q_norm").get(head_dim, "weight")?,
                k_norm: a.pp("k_norm").get(head_dim, "weight")?,
                in_norm: l.pp("input_layernorm").get(hidden, "weight")?,
                post_attn_norm: l.pp("post_attention_layernorm").get(hidden, "weight")?,
                pre_ff_norm: l.pp("pre_feedforward_layernorm").get(hidden, "weight")?,
                post_ff_norm: l.pp("post_feedforward_layernorm").get(hidden, "weight")?,
                gate: lin(l.pp("mlp").pp("gate_proj"), inter, hidden)?,
                up: lin(l.pp("mlp").pp("up_proj"), inter, hidden)?,
                down: lin(l.pp("mlp").pp("down_proj"), hidden, inter)?,
                sliding: types
                    .get(i)
                    .map(|t| t == "sliding_attention")
                    .unwrap_or((i + 1) % 6 != 0),
            });
        }
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let id = dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or("gemma3".into());
        let eos = [tok.token_to_id("<end_of_turn>"), tok.token_to_id("<eos>")]
            .into_iter()
            .flatten()
            .collect();
        Ok(Gemma3 {
            spec: Spec {
                id,
                layers,
                hidden,
                heads,
                kv_heads,
                head_dim,
                vocab,
                bos: tok.token_to_id("<bos>").unwrap_or(2),
                eos,
            },
            tok,
            embed: m.pp("embed_tokens").get((vocab, hidden), "weight")?,
            layers: ls,
            norm: m.pp("norm").get(hidden, "weight")?,
            eps: f("rms_norm_eps", 1e-6),
            window: u("sliding_window").unwrap_or(512),
            rope_local: f("rope_local_base_freq", 10_000.0),
            rope_global: f("rope_theta", 1_000_000.0),
            scale: 1.0 / f("query_pre_attn_scalar", head_dim as f64).sqrt(),
            dev,
        })
    }

    fn rope(&self, x: &Tensor, pos0: usize, base: f64) -> Result<Tensor> {
        // x: [h, n, d] → rotate halves (the non-interleaved layout)
        let (_, n, d) = x.dims3()?;
        let inv: Vec<f32> = (0..d / 2)
            .map(|i| (1.0 / base.powf(2.0 * i as f64 / d as f64)) as f32)
            .collect();
        let pos: Vec<f32> = (pos0..pos0 + n).map(|p| p as f32).collect();
        let freqs = Tensor::new(pos, &self.dev)?
            .unsqueeze(1)?
            .matmul(&Tensor::new(inv, &self.dev)?.unsqueeze(0)?)?; // [n, d/2]
        let (cos, sin) = (freqs.cos()?.contiguous()?, freqs.sin()?.contiguous()?);
        Ok(candle_nn::rotary_emb::rope(&x.unsqueeze(0)?.contiguous()?, &cos, &sin)?.squeeze(0)?)
    }
}

impl Model for Gemma3 {
    fn spec(&self) -> &Spec {
        &self.spec
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let t = Tensor::new(ids, &self.dev)?;
        Ok((self.embed.index_select(&t, 0)? * (self.spec.hidden as f64).sqrt())?)
    }

    fn attention(
        &self,
        i: usize,
        x: &Tensor,
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<(Tensor, Tensor)> {
        let l = &self.layers[i];
        let s = &self.spec;
        let n = x.dims()[0];
        let h = rms(x, &l.in_norm, self.eps)?;
        let heads = |t: Tensor, k: usize| -> Result<Tensor> {
            Ok(t.reshape((n, k, s.head_dim))?
                .transpose(0, 1)?
                .contiguous()?)
        };
        let q = rms(&heads(l.q.forward(&h)?, s.heads)?, &l.q_norm, self.eps)?;
        let k = rms(&heads(l.k.forward(&h)?, s.kv_heads)?, &l.k_norm, self.eps)?;
        let v = heads(l.v.forward(&h)?, s.kv_heads)?;
        let base = if l.sliding {
            self.rope_local
        } else {
            self.rope_global
        };
        let (q, k) = (self.rope(&q, pos0, base)?, self.rope(&k, pos0, base)?);
        let (ka, va) = cache.append(i, k, v)?;
        let total = ka.dims()[1];
        // grouped heads: each KV head serves heads / kv_heads query heads
        let group = s.heads / s.kv_heads;
        let qg = q.reshape((s.kv_heads, group * n, s.head_dim))?;
        let scores = (qg.matmul(&ka.transpose(1, 2)?.contiguous()?)? * self.scale)?
            .reshape((s.heads, n, total))?;
        // causal (and sliding-window) mask
        let past = total - n;
        let mask: Vec<f32> = (0..n)
            .flat_map(|r| {
                let p = past + r;
                (0..total).map(move |c| {
                    if c > p || (l.sliding && p - c >= self.window) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, n, total), &self.dev)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        Ok((probs, va))
    }

    fn attention_out(&self, i: usize, probs: &Tensor, values: &Tensor) -> Result<Tensor> {
        let l = &self.layers[i];
        let s = &self.spec;
        let (_, n, total) = probs.dims3()?;
        let group = s.heads / s.kv_heads;
        let pg = probs.reshape((s.kv_heads, group * n, total))?;
        let out = pg
            .matmul(values)?
            .reshape((s.heads, n, s.head_dim))?
            .transpose(0, 1)?
            .reshape((n, s.heads * s.head_dim))?;
        rms(&l.o.forward(&out)?, &l.post_attn_norm, self.eps)
    }

    fn mlp(&self, i: usize, mid: &Tensor) -> Result<Tensor> {
        let l = &self.layers[i];
        let h = rms(mid, &l.pre_ff_norm, self.eps)?;
        let g = l.gate.forward(&h)?.gelu()?; // tanh approximation, as gelu_pytorch_tanh
        let y = l.down.forward(&(g * l.up.forward(&h)?)?)?;
        rms(&y, &l.post_ff_norm, self.eps)
    }

    fn final_norm(&self, x: &Tensor) -> Result<Tensor> {
        rms(x, &self.norm, self.eps)
    }

    fn unembed(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.embed.t()?)?)
    }
}

/// Load a model from the temp store by id (e.g. "gemma-3-270m-it").
pub fn load(id: &str) -> Result<Gemma3> {
    let d = latent_core::store::model_dir(id);
    if !d.join("model.safetensors").exists() {
        return err(format!(
            "{} missing: `latent models fetch {id}`",
            d.display()
        ));
    }
    Gemma3::load(&d)
}

/// Gemma's chat turn around a user message, ready for the model's reply.
pub fn chat(user: &str) -> String {
    format!("<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
}
