//! # latent-algos — algorithms in latent space
//!
//! Each one is a few lines on [`Run`]: what it reads, where, and what it does
//! with the vectors.
//!
//! | algorithm | reads / writes | gives |
//! |---|---|---|
//! | [`generate`] | logits, hook at every site | text |
//! | [`lens`] | the stream after each layer, through the final norm and unembedding | what each layer would say next |
//! | [`readout`] | the stream at any site, pooled (mean / last token, optionally framed by a template), **stopping there** | a [`Vector`] tagged with its space |
//! | [`Bank`] | labelled readouts | nearest-example decisions ([`Choice`]) |
//! | [`choose`] | log-probabilities of each label as the answer, one shared prefix forked per label | a [`Choice`], optionally calibrated |
//! | [`direction`] + [`steer`] | the mean difference of two sets of readouts, added to the stream while generating | steered text |
//! | [`Tagger`] | per-token vectors at a site, prototypes from example spans | extracted spans |
//!
//! A decision read at layer k of L costs k/L of a forward pass: [`readout`]
//! stops the run there, and [`Run::layer_steps`] counts what was spent.

use candle_core::{Tensor, D};
use latent_core::{err, Choice, Model, Pool, Result, Run, Site, Space, Vector};

/// Per layer: the site and its top next tokens with probabilities.
pub type LensRows = Vec<(String, Vec<(String, f32)>)>;
/// Per token: its vector, and its character span in the text.
type TokenVectors = (Vec<Vec<f32>>, Vec<(usize, usize)>);

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Greedy reply to a prompt (already in the model's chat format). `hook` sees
/// (and may change) every site of every step.
pub fn generate(
    m: &dyn Model,
    prompt: &str,
    max: usize,
    mut hook: impl FnMut(Site, &mut Tensor) -> Result<()>,
) -> Result<String> {
    let mut run = Run::new(m);
    run.feed(&m.encode(prompt)?);
    let mut out = vec![];
    for _ in 0..max {
        let logits = run.finish_with(&mut hook)?;
        let next = argmax(&Run::log_softmax_last(&logits)?) as u32;
        if m.spec().eos.contains(&next) {
            break;
        }
        out.push(next);
        run.feed(&[next]);
    }
    m.decode(&out)
}

/// What the model would say next if it stopped after each layer: the last
/// token's stream after layer i, through the final norm and unembedding.
pub fn lens(m: &dyn Model, text: &str, top: usize) -> Result<LensRows> {
    let mut run = Run::new(m);
    run.feed(&m.encode(text)?);
    let mut out = vec![];
    while let Some(s) = run.step()? {
        if let Site::Layer(_, latent_core::Point::Post) = s {
            let x = run.get();
            let last = x.get(x.dims()[0] - 1)?.unsqueeze(0)?;
            let p = candle_nn::ops::softmax_last_dim(&m.unembed(&m.final_norm(&last)?)?)?
                .squeeze(0)?
                .to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.select_nth_unstable_by(top, |a, b| p[*b].total_cmp(&p[*a]));
            let mut best: Vec<(String, f32)> = idx[..top]
                .iter()
                .map(|i| (m.piece(*i as u32), p[*i]))
                .collect();
            best.sort_by(|a, b| b.1.total_cmp(&a.1));
            out.push((s.to_string(), best));
        }
        if s == Site::Norm {
            break;
        }
    }
    Ok(out)
}

/// How to read one vector for a text.
#[derive(Clone, Debug)]
pub struct Readout {
    pub site: Site,
    pub pool: Pool,
    /// frames the text: `{}` is replaced by it (e.g. a prompt that makes the
    /// last token sum the text up)
    pub template: Option<String>,
}

impl Readout {
    pub fn new(site: Site, pool: Pool) -> Self {
        Readout {
            site,
            pool,
            template: None,
        }
    }
    pub fn framed(mut self, t: &str) -> Self {
        self.template = Some(t.into());
        self
    }
    pub fn space(&self, m: &dyn Model) -> Space {
        Space {
            model: m.spec().id.clone(),
            site: self.site,
            pool: self.pool,
            template: self.template.clone(),
        }
    }
}

/// Read one vector for `text`: run to the site and **stop** (later layers never
/// run). Returns the vector and the layer steps spent.
pub fn readout(m: &dyn Model, text: &str, r: &Readout) -> Result<(Vector, usize)> {
    let full = match &r.template {
        Some(t) => t.replace("{}", text),
        None => format!("<bos>{text}"),
    };
    let mut run = Run::new(m);
    run.feed(&m.encode(&full)?);
    let x = run.run_to(r.site)?.clone();
    let spent = run.layer_steps;
    run.stop()?;
    let n = x.dims()[0];
    let v = match r.pool {
        Pool::Last => x.get(n - 1)?,
        Pool::Mean if n > 1 => x.narrow(0, 1, n - 1)?.mean(0)?, // skip <bos>
        Pool::Mean => x.mean(0)?,
    };
    Ok((
        Vector {
            data: v.to_vec1::<f32>()?,
            space: r.space(m),
        },
        spent,
    ))
}

/// Labelled vectors read one way; decides by the nearest example. `center`
/// subtracts the bank's mean first: the residual stream has a few huge
/// shared directions that otherwise dominate every cosine.
pub struct Bank {
    pub readout: Readout,
    pub items: Vec<(String, Vector)>,
    pub center: bool,
    mean: Vec<f32>,
    /// layer steps spent reading the examples
    pub spent: usize,
}

impl Bank {
    pub fn build(
        m: &dyn Model,
        readout_how: Readout,
        examples: &[(&str, Vec<&str>)],
        center: bool,
    ) -> Result<Bank> {
        let mut items = vec![];
        let mut spent = 0;
        for (label, exs) in examples {
            for e in exs {
                let (v, s) = readout(m, e, &readout_how)?;
                spent += s;
                items.push((label.to_string(), v));
            }
        }
        let dim = items.first().map(|i| i.1.dim()).unwrap_or(0);
        let mut mean = vec![0f32; dim];
        if center {
            for (_, v) in &items {
                mean.iter_mut()
                    .zip(&v.data)
                    .for_each(|(a, b)| *a += b / items.len() as f32);
            }
        }
        Ok(Bank {
            readout: readout_how,
            items,
            center,
            mean,
            spent,
        })
    }

    fn c(&self, v: &[f32]) -> Vec<f32> {
        v.iter().zip(&self.mean).map(|(a, b)| a - b).collect()
    }

    /// Nearest example per label (cosine), as a Choice.
    pub fn nearest(&self, v: &Vector) -> Result<Choice> {
        if let Some((_, first)) = self.items.first() {
            if v.space != first.space {
                return err(format!(
                    "vector from {} asked of a bank read at {}",
                    v.space, first.space
                ));
            }
        }
        let q = self.c(&v.data);
        let mut best: Vec<(String, f32)> = vec![];
        for (l, e) in &self.items {
            let s = latent_core::vector::cosine(&q, &self.c(&e.data));
            match best.iter_mut().find(|b| b.0 == *l) {
                Some(b) => b.1 = b.1.max(s),
                None => best.push((l.clone(), s)),
            }
        }
        Choice::from_scores(best, format!("nearest@{}", v.space))
    }

    /// Decide for a text: read it (stopping at the bank's site) and find the nearest.
    pub fn decide(&self, m: &dyn Model, text: &str) -> Result<(Choice, usize)> {
        let (v, s) = readout(m, text, &self.readout)?;
        Ok((self.nearest(&v)?, s))
    }
}

/// Constrained choice: the probability of each label as the continuation of
/// `head` (tokenised together, so a label's first piece is the one the model
/// would write there). The shared tokens are read once; each label forks the
/// run. With `null_head` (a content-free version of the prompt), scores are
/// divided by the scores there: calibration against name / position bias.
pub fn choose(
    m: &dyn Model,
    head: &str,
    labels: &[String],
    tail: &str,
    null_head: Option<&str>,
) -> Result<Choice> {
    let score = |head: &str| -> Result<Vec<f32>> {
        let seqs: Vec<Vec<u32>> = labels
            .iter()
            .map(|l| m.encode(&format!("{head}{l}{tail}")))
            .collect::<Result<_>>()?;
        let mut lcp = seqs[0].len();
        for s in &seqs[1..] {
            lcp = lcp.min(seqs[0].iter().zip(s).take_while(|(a, b)| a == b).count());
        }
        lcp = lcp
            .min(seqs.iter().map(|s| s.len() - 1).min().unwrap_or(0))
            .max(1);
        let mut base = Run::new(m);
        base.feed(&seqs[0][..lcp]);
        let first = Run::log_softmax_last(&base.finish()?)?;
        let mut out = vec![];
        for s in &seqs {
            let t = &s[lcp..];
            let mut lp = first[t[0] as usize];
            if t.len() > 1 {
                let mut r = base.fork();
                r.feed(&t[..t.len() - 1]);
                let logits = r.finish()?;
                let rows = candle_nn::ops::log_softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;
                for (row, next) in rows.iter().zip(&t[1..]) {
                    lp += row[*next as usize];
                }
            }
            out.push(lp);
        }
        Ok(out)
    };
    let mut lp = score(head)?;
    if let Some(n) = null_head {
        let null = score(n)?;
        lp.iter_mut().zip(&null).for_each(|(a, b)| *a -= b);
    }
    let mx = lp.iter().cloned().fold(f32::MIN, f32::max);
    let z: f32 = lp.iter().map(|x| (x - mx).exp()).sum();
    Choice::from_scores(
        labels
            .iter()
            .cloned()
            .zip(lp.iter().map(|x| (x - mx).exp() / z))
            .collect(),
        if null_head.is_some() {
            "choose+calibrated"
        } else {
            "choose"
        },
    )
}

/// A direction in the stream at `site`: mean(positive readouts) − mean(negative).
pub fn direction(m: &dyn Model, site: Site, pos: &[&str], neg: &[&str]) -> Result<Tensor> {
    let r = Readout::new(site, Pool::Mean);
    let mean = |xs: &[&str]| -> Result<Vec<f32>> {
        let mut acc: Vec<f32> = vec![];
        for x in xs {
            let v = readout(m, x, &r)?.0.data;
            if acc.is_empty() {
                acc = vec![0.0; v.len()];
            }
            acc.iter_mut()
                .zip(&v)
                .for_each(|(a, b)| *a += b / xs.len() as f32);
        }
        Ok(acc)
    };
    let (p, n) = (mean(pos)?, mean(neg)?);
    Ok(Tensor::new(
        p.iter().zip(&n).map(|(a, b)| a - b).collect::<Vec<f32>>(),
        &candle_core::Device::Cpu,
    )?)
}

/// A hook that adds `alpha · dir` to the stream at `site` on every token.
pub fn steer(site: Site, dir: Tensor, alpha: f64) -> impl FnMut(Site, &mut Tensor) -> Result<()> {
    move |s, t| {
        if s == site {
            *t = t.broadcast_add(&(&dir * alpha)?)?;
        }
        Ok(())
    }
}

/// Token vectors at `site` for `text` and each token's character span in it.
fn token_vectors(m: &dyn Model, text: &str, site: Site) -> Result<TokenVectors> {
    let enc = m
        .tokenizer()
        .encode(text, false)
        .map_err(|e| latent_core::Error::Tokenizer(e.to_string()))?;
    let mut ids = vec![m.spec().bos];
    ids.extend(enc.get_ids());
    let mut run = Run::new(m);
    run.feed(&ids);
    let x = run.run_to(site)?.to_vec2::<f32>()?;
    run.stop()?;
    Ok((x[1..].to_vec(), enc.get_offsets().to_vec()))
}

/// Token-level extraction: each type is a prototype (the mean vector of its
/// example spans' tokens, read at `site` in their sentences); `none` is the
/// mean of the other tokens. Every token of a new text takes the nearest
/// prototype; runs of one type are its spans.
pub struct Tagger {
    pub site: Site,
    pub protos: Vec<(String, Vec<f32>)>,
    /// every example token (type or `none`), centred: for nearest-token tagging
    pub tokens: Vec<(String, Vec<f32>)>,
    /// tag each token by its nearest example token (true) or nearest prototype (false)
    pub nearest_token: bool,
    mean: Vec<f32>,
}

impl Tagger {
    /// `examples`: (type, sentence, the span of it that is that type).
    pub fn build(m: &dyn Model, site: Site, examples: &[(&str, &str, &str)]) -> Result<Tagger> {
        let mut sums: Vec<(String, Vec<f32>, usize)> = vec![];
        let mut all: Vec<Vec<f32>> = vec![];
        let mut labelled: Vec<(String, Vec<f32>)> = vec![];
        for (ty, sent, span) in examples {
            let (vs, offs) = token_vectors(m, sent, site)?;
            let Some(at) = sent.find(span) else {
                return err(format!("“{span}” is not in “{sent}”"));
            };
            let (a, b) = (at, at + span.len());
            for (v, (s, e)) in vs.iter().zip(&offs) {
                let key = if *s < b && *e > a {
                    ty.to_string()
                } else {
                    "none".to_string()
                };
                match sums.iter_mut().find(|x| x.0 == key) {
                    Some(x) => {
                        x.1.iter_mut().zip(v).for_each(|(p, q)| *p += q);
                        x.2 += 1;
                    }
                    None => sums.push((key.clone(), v.clone(), 1)),
                }
                all.push(v.clone());
                labelled.push((key, v.clone()));
            }
        }
        let dim = all.first().map(|v| v.len()).unwrap_or(0);
        let mut mean = vec![0f32; dim];
        for v in &all {
            mean.iter_mut()
                .zip(v)
                .for_each(|(a, b)| *a += b / all.len() as f32);
        }
        let protos = sums
            .into_iter()
            .map(|(k, s, n)| {
                (
                    k,
                    s.iter()
                        .zip(&mean)
                        .map(|(x, mu)| x / n as f32 - mu)
                        .collect(),
                )
            })
            .collect();
        let tokens = labelled
            .into_iter()
            .map(|(k, v)| (k, v.iter().zip(&mean).map(|(x, mu)| x - mu).collect()))
            .collect();
        Ok(Tagger {
            site,
            protos,
            tokens,
            nearest_token: true,
            mean,
        })
    }

    /// The spans of each type in `text`: (type, text, mean similarity).
    pub fn tag(&self, m: &dyn Model, text: &str) -> Result<Vec<(String, String, f32)>> {
        let (vs, offs) = token_vectors(m, text, self.site)?;
        let labels: Vec<(String, f32)> = vs
            .iter()
            .map(|v| {
                let c: Vec<f32> = v.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
                let pool = if self.nearest_token {
                    &self.tokens
                } else {
                    &self.protos
                };
                pool.iter()
                    .map(|(k, p)| (k.clone(), latent_core::vector::cosine(&c, p)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap()
            })
            .collect();
        let mut out: Vec<(String, String, f32)> = vec![];
        let mut i = 0;
        while i < labels.len() {
            let ty = labels[i].0.clone();
            let mut j = i;
            let mut s = 0.0;
            while j < labels.len() && labels[j].0 == ty {
                s += labels[j].1;
                j += 1;
            }
            if ty != "none" {
                let (a, b) = (offs[i].0, offs[j - 1].1);
                out.push((ty, text[a..b].trim().to_string(), s / (j - i) as f32));
            }
            i = j;
        }
        Ok(out)
    }
}
