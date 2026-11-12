//! Where in the forward pass. Names follow the residual-stream picture:
//! each layer reads the stream (`Pre`), attends (`Attn` probabilities,
//! `AttnOut` its contribution), adds it (`Mid`), runs the MLP (`MlpOut`) and
//! adds that (`Post`). `Post` of layer i is `Pre` of layer i+1: the same
//! tensor, so changing either changes what layer i+1 reads.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Point {
    /// the residual stream entering the layer: [tokens, hidden]
    Pre,
    /// attention probabilities: [heads, tokens, all tokens so far]
    Attn,
    /// what attention adds to the stream (after the output projection and any norm): [tokens, hidden]
    AttnOut,
    /// the stream after attention: [tokens, hidden]
    Mid,
    /// what the MLP adds: [tokens, hidden]
    MlpOut,
    /// the stream leaving the layer: [tokens, hidden]
    Post,
}

pub const POINTS: [Point; 6] = [
    Point::Pre,
    Point::Attn,
    Point::AttnOut,
    Point::Mid,
    Point::MlpOut,
    Point::Post,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Site {
    /// token embeddings (already scaled, as the first layer reads them)
    Embed,
    Layer(usize, Point),
    /// the stream after the final norm
    Norm,
    /// next-token logits: [tokens, vocab]
    Logits,
}

impl Site {
    /// The site after this one in a model with `layers` layers.
    pub fn next(self, layers: usize) -> Option<Site> {
        Some(match self {
            Site::Embed => Site::Layer(0, Point::Pre),
            Site::Layer(i, p) => match POINTS.iter().position(|x| *x == p).unwrap() {
                5 if i + 1 < layers => Site::Layer(i + 1, Point::Pre),
                5 => Site::Norm,
                k => Site::Layer(i, POINTS[k + 1]),
            },
            Site::Norm => Site::Logits,
            Site::Logits => return None,
        })
    }

    /// Layers fully run once this site is reached (for compute accounting).
    pub fn layers_done(self, layers: usize) -> usize {
        match self {
            Site::Embed => 0,
            Site::Layer(i, Point::Post) => i + 1,
            Site::Layer(i, _) => i,
            _ => layers,
        }
    }

    /// The stream after layer `i` (0-based); `post(layers-1)` is just before the final norm.
    pub fn post(i: usize) -> Site {
        Site::Layer(i, Point::Post)
    }
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Site::Embed => write!(f, "embed"),
            Site::Layer(i, p) => write!(f, "L{i}.{}", format!("{p:?}").to_lowercase()),
            Site::Norm => write!(f, "norm"),
            Site::Logits => write!(f, "logits"),
        }
    }
}
