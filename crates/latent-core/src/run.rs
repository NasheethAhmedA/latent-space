//! A forward pass you step through.
//!
//! ```ignore
//! let mut run = Run::new(&model);
//! run.feed(&model.encode("The capital of France is")?);
//! run.run_to(Site::post(5))?;            // pause after layer 5
//! let h = run.get().clone();             // read the stream there
//! run.set(h.broadcast_add(&steer)?)?;    // change it
//! let logits = run.finish()?;            // carry on to the end
//! let other = run.fork();                // a second continuation of the same prefix
//! ```
//!
//! The value at a site is what the next stage reads: setting it at `Pre`,
//! `Mid`, `Post` or `Embed` changes the residual stream; at `Attn` the
//! attention pattern; at `AttnOut` / `MlpOut` what that block adds; at `Norm`
//! what the unembedding reads.

use crate::{err, Cache, Model, Point, Result, Site, Tensor};

#[derive(Clone)]
pub struct Run<'m> {
    m: &'m dyn Model,
    pub cache: Cache,
    /// every token this run has committed or is processing
    pub tokens: Vec<u32>,
    pending: Vec<u32>,
    /// where the step stands (None: before `Embed`, or between steps)
    site: Option<Site>,
    /// the residual stream of the step in flight
    x: Option<Tensor>,
    /// the value at the current site
    cur: Option<Tensor>,
    values: Option<Tensor>,
    /// stopped before the end of a step: the cache is incomplete past this layer
    broken: bool,
    /// layer forward passes executed, over all tokens (a compute meter)
    pub layer_steps: usize,
}

impl<'m> Run<'m> {
    pub fn new(m: &'m dyn Model) -> Self {
        Run {
            m,
            cache: Cache::new(m.spec().layers),
            tokens: vec![],
            pending: vec![],
            site: None,
            x: None,
            cur: None,
            values: None,
            broken: false,
            layer_steps: 0,
        }
    }

    pub fn model(&self) -> &'m dyn Model {
        self.m
    }

    /// Queue tokens for the next step (they continue whatever was committed).
    pub fn feed(&mut self, ids: &[u32]) -> &mut Self {
        self.pending.extend_from_slice(ids);
        self
    }

    /// The site the run is paused at.
    pub fn site(&self) -> Option<Site> {
        self.site
    }

    /// The tensor at the paused site.
    pub fn get(&self) -> &Tensor {
        self.cur
            .as_ref()
            .expect("run is not paused at a site (call step / run_to first)")
    }

    /// Replace the tensor at the paused site; the rest of the pass reads the new one.
    pub fn set(&mut self, t: Tensor) -> Result<()> {
        let Some(site) = self.site else {
            return err("not paused at a site");
        };
        if t.dims() != self.get().dims() {
            return err(format!(
                "at {site}: shape {:?}, expected {:?}",
                t.dims(),
                self.get().dims()
            ));
        }
        if matches!(
            site,
            Site::Embed | Site::Layer(_, Point::Pre | Point::Mid | Point::Post)
        ) {
            self.x = Some(t.clone());
        }
        self.cur = Some(t);
        Ok(())
    }

    /// Advance to the next site. Returns it, or None when the step is done
    /// (after `Logits`; the tokens are then committed to the cache).
    pub fn step(&mut self) -> Result<Option<Site>> {
        let layers = self.m.spec().layers;
        let next = match self.site {
            None => {
                if self.broken {
                    return err("this run stopped part-way through a step; fork before stopping early, or start a new run");
                }
                if self.pending.is_empty() {
                    return Ok(None);
                }
                let e = self.m.embed(&self.pending)?;
                self.x = Some(e.clone());
                self.cur = Some(e);
                Site::Embed
            }
            Some(Site::Logits) => {
                self.cache.len += self.pending.len();
                self.tokens.append(&mut self.pending);
                self.site = None;
                self.cur = None;
                self.x = None;
                return Ok(None);
            }
            Some(s) => {
                let n = s.next(layers).unwrap();
                let x = self.x.clone().unwrap();
                self.cur = Some(match n {
                    Site::Layer(_, Point::Pre) => x,
                    Site::Layer(i, Point::Attn) => {
                        let (p, v) = self.m.attention(i, &x, self.cache.len, &mut self.cache)?;
                        self.values = Some(v);
                        p
                    }
                    Site::Layer(i, Point::AttnOut) => {
                        self.m
                            .attention_out(i, self.get(), self.values.as_ref().unwrap())?
                    }
                    Site::Layer(_, Point::Mid) | Site::Layer(_, Point::Post) => {
                        let nx = (x + self.get())?;
                        self.x = Some(nx.clone());
                        if matches!(n, Site::Layer(_, Point::Post)) {
                            self.layer_steps += self.pending.len();
                        }
                        nx
                    }
                    Site::Layer(i, Point::MlpOut) => self.m.mlp(i, &x)?,
                    Site::Norm => self.m.final_norm(&x)?,
                    Site::Logits => self.m.unembed(self.get())?,
                    Site::Embed => unreachable!(),
                });
                n
            }
        };
        self.site = Some(next);
        Ok(Some(next))
    }

    /// Step until paused at `site` (in this step). Errors if it was already passed.
    pub fn run_to(&mut self, site: Site) -> Result<&Tensor> {
        if let Some(s) = self.site {
            if s > site {
                return err(format!("{site} was already passed (at {s})"));
            }
        }
        loop {
            if self.site == Some(site) {
                return Ok(self.get());
            }
            if self.step()?.is_none() {
                return err(format!("{site} not reached"));
            }
        }
    }

    /// Run the rest of the step; returns the logits ([n, vocab]) and commits the tokens.
    pub fn finish(&mut self) -> Result<Tensor> {
        self.finish_with(|_, _| Ok(()))
    }

    /// As [`finish`](Self::finish), calling `hook` at every site with the tensor
    /// there (it may change it).
    pub fn finish_with(
        &mut self,
        mut hook: impl FnMut(Site, &mut Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        loop {
            match self.step()? {
                None => break,
                Some(s) => {
                    let mut t = self.get().clone();
                    let before = t.clone();
                    hook(s, &mut t)?;
                    if !same(&t, &before) {
                        self.set(t)?;
                    }
                    if s == Site::Logits {
                        let l = self.get().clone();
                        self.step()?; // commit
                        return Ok(l);
                    }
                }
            }
        }
        err("nothing to run (feed tokens first)")
    }

    /// Stop here for good: the step won't complete (an early exit). Returns the
    /// tensor at the paused site; the run can't continue, but forks taken
    /// earlier can.
    pub fn stop(&mut self) -> Result<Tensor> {
        let t = self.get().clone();
        if self.site.is_some() && self.site != Some(Site::Logits) {
            self.broken = true;
        }
        self.site = None;
        Ok(t)
    }

    /// An independent copy (cache shared until either side appends).
    pub fn fork(&self) -> Run<'m> {
        self.clone()
    }

    /// Next-token log-probabilities after the last committed token, from `logits` rows.
    pub fn log_softmax_last(logits: &Tensor) -> Result<Vec<f32>> {
        let n = logits.dims()[0];
        let last = logits.get(n - 1)?;
        Ok(candle_nn::ops::log_softmax(&last, 0)?.to_vec1::<f32>()?)
    }
}

fn same(a: &Tensor, b: &Tensor) -> bool {
    // cheap identity test: the hook either returned the tensor it got or a new one
    a.id() == b.id()
}
