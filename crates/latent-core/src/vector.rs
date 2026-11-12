//! Readouts and decisions.

use crate::{err, Result, Site};
use std::fmt;

/// How token vectors at a site become one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pool {
    /// the mean over the text's own tokens
    Mean,
    /// the last token (it has attended to everything before it)
    Last,
}

/// Where a vector was read: model, site and pooling (and the template that
/// framed the text, if any). Vectors compare only within one space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Space {
    pub model: String,
    pub site: Site,
    pub pool: Pool,
    pub template: Option<String>,
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}:{:?}", self.model, self.site, self.pool)?;
        if self.template.is_some() {
            write!(f, "+template")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Vector {
    pub data: Vec<f32>,
    pub space: Space,
}

impl Vector {
    pub fn cosine(&self, o: &Vector) -> Result<f32> {
        if self.space != o.space {
            return err(format!(
                "can't compare a vector from {} with one from {}",
                self.space, o.space
            ));
        }
        Ok(cosine(&self.data, &o.data))
    }
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// One of N, with how sure: every decider's answer.
#[derive(Clone, Debug)]
pub struct Choice {
    pub label: String,
    pub score: f32,
    /// best score minus the second best
    pub margin: f32,
    /// every label with its score, best first
    pub alternatives: Vec<(String, f32)>,
    /// how it was decided
    pub source: String,
}

impl Choice {
    pub fn from_scores(
        mut scored: Vec<(String, f32)>,
        source: impl Into<String>,
    ) -> Result<Choice> {
        if scored.is_empty() {
            return err("no options");
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let margin = scored[0].1 - scored.get(1).map(|x| x.1).unwrap_or(0.0);
        Ok(Choice {
            label: scored[0].0.clone(),
            score: scored[0].1,
            margin,
            alternatives: scored,
            source: source.into(),
        })
    }
}
