//! The stepped run must compute exactly what a plain forward pass does, and
//! forks, early stops and edits must behave as documented.
use latent_core::{Model, Point, Run, Site};

fn model() -> Option<latent_gemma::Gemma3> {
    latent_gemma::load("gemma-3-270m-it")
        .map_err(|e| eprintln!("skipped: {e}"))
        .ok()
}

fn last_row(t: &candle_core::Tensor) -> Vec<f32> {
    t.get(t.dims()[0] - 1).unwrap().to_vec1::<f32>().unwrap()
}

fn close(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn incremental_forked_and_whole_runs_agree() {
    let Some(m) = model() else { return };
    let ids = m
        .encode("<bos>The capital of France is Paris, and the capital of Italy is")
        .unwrap();
    // whole prompt at once
    let mut whole = Run::new(&m);
    whole.feed(&ids);
    let a = last_row(&whole.finish().unwrap());
    // the same, fed in two parts through a fork
    let mut first = Run::new(&m);
    first.feed(&ids[..5]);
    first.finish().unwrap();
    let mut rest = first.fork();
    rest.feed(&ids[5..]);
    let b = last_row(&rest.finish().unwrap());
    assert!(
        close(&a, &b) < 1e-2,
        "incremental vs whole: {}",
        close(&a, &b)
    );
    // the fork left the original untouched
    assert_eq!(first.cache.len, 5);
    assert_eq!(rest.cache.len, ids.len());
    // and the next word is Rome
    let best = a
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .unwrap()
        .0 as u32;
    assert_eq!(m.piece(best).trim(), "Rome");
}

#[test]
fn sites_come_in_order_and_edits_take_effect() {
    let Some(m) = model() else { return };
    let ids = m.encode("<bos>Hello there").unwrap();
    let mut run = Run::new(&m);
    run.feed(&ids);
    let mut seen = vec![];
    while let Some(s) = run.step().unwrap() {
        seen.push(s);
    }
    assert_eq!(seen.len(), 1 + 6 * m.spec().layers + 2);
    assert_eq!(seen[0], Site::Embed);
    assert_eq!(seen[1], Site::Layer(0, Point::Pre));
    assert_eq!(*seen.last().unwrap(), Site::Logits);
    // the attention pattern is a distribution over the tokens so far
    let mut r = Run::new(&m);
    r.feed(&ids);
    let p = r.run_to(Site::Layer(3, Point::Attn)).unwrap().clone();
    assert_eq!(p.dims(), &[m.spec().heads, ids.len(), ids.len()]);
    let sums = p
        .sum(2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-4));
    // zeroing the stream after layer 5 changes the answer; an early stop spends 6 of 18 layers
    let base = {
        let mut x = Run::new(&m);
        x.feed(&ids);
        last_row(&x.finish().unwrap())
    };
    let mut e = Run::new(&m);
    e.feed(&ids);
    let h = e.run_to(Site::post(5)).unwrap().zeros_like().unwrap();
    e.set(h).unwrap();
    let edited = last_row(&e.finish().unwrap());
    assert!(close(&base, &edited) > 1.0);
    let mut early = Run::new(&m);
    early.feed(&ids);
    early.run_to(Site::post(5)).unwrap();
    assert_eq!(early.layer_steps, 6 * ids.len());
    early.stop().unwrap();
    assert!(
        early.feed(&ids).step().is_err(),
        "a stopped run can't carry on"
    );
}
