//! # latent-core — latent space programming
//!
//! Programs have used language models as a black box: text in, text out.
//! The useful part of a model is inside it: the vectors it builds at every
//! layer for every token. Embeddings, constrained choice, probes, steering,
//! manifold projection and extraction are all algorithms that read (or
//! write) those vectors. This crate gives them one foundation:
//!
//! - [`Site`]: a named place in the forward pass (`Embed`, per layer `Pre`,
//!   `Attn` (the attention pattern), `AttnOut`, `Mid`, `MlpOut`, `Post`, then
//!   `Norm`, `Logits`).
//! - [`Model`]: an architecture split into those stages, so nothing is fused
//!   away.
//! - [`Run`]: a forward pass you **step**: pause at any site, read the tensor
//!   there, replace it, carry on; stop early (spend part of the compute);
//!   **fork** to try several continuations of one shared prefix.
//! - [`Cache`]: the KV cache as a value: cloned, truncated, shared.
//! - [`Vector`]: a readout that remembers where it was read (model, site,
//!   pooling), so vectors from different places can't be mixed by mistake.
//! - [`Choice`]: one of N with scores and a margin, the shape every decider
//!   returns.
//!
//! Model weights live only in a temporary store ([`store`]), cleared in one
//! step.

pub mod cache;
pub mod model;
pub mod run;
pub mod site;
pub mod store;
pub mod vector;

pub use cache::Cache;
pub use candle_core::{DType, Device, Tensor};
pub use model::{Model, Spec};
pub use run::Run;
pub use site::{Point, Site};
pub use vector::{Choice, Pool, Space, Vector};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tensor: {0}")]
    Tensor(#[from] candle_core::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn err<T>(m: impl Into<String>) -> Result<T> {
    Err(Error::Msg(m.into()))
}
