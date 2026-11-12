# latent-space

**A Rust framework for latent space programming:** writing algorithms that run
*inside* a language model's forward pass, not just on its text output.

Most software treats a language model as a function from text to text. Most of
what makes these models useful is in the vectors they compute along the way:
the residual stream at every layer, for every token, and the attention patterns
that mix them. Many techniques that make LLMs fast and controllable are
algorithms over those vectors:

| technique | what it really does |
|---|---|
| sentence embeddings | pool the hidden states of some layer |
| constrained decisions | compare the logits of a fixed set of answers |
| probes and early exit | decide from layer *k* of *L*, spending *k/L* of the compute |
| activation steering | add a direction to the residual stream |
| subspace / manifold projection | compare vectors along chosen directions only |
| entity and slot extraction | compare per-token vectors with examples |
| logit lens | read what each layer would predict next |

`latent-space` gives these one foundation: a forward pass you can **pause, read,
alter, fork and stop early at every step**, plus data types that keep track of
where each vector came from. These algorithms become small programs instead of
one-off research scripts, so they're easy to combine and compare, and new ones
are easy to try.

## The idea: a forward pass you step through

```rust
use latent_core::{Model, Run, Site};

let model = latent_gemma::load("gemma-3-270m-it")?;
let mut run = Run::new(&model);
run.feed(&model.encode("<bos>The capital of France is")?);

run.run_to(Site::post(5))?;              // pause after layer 5
let h = run.get().clone();               // the residual stream: [tokens, hidden]
run.set(h.broadcast_add(&direction)?)?;  // change it
let logits = run.finish()?;              // the rest of the pass reads the new stream

let branch = run.fork();                 // another continuation of the same prefix (KV cache shared)
```

Every forward step visits these **sites** in order, and each one can be read
and replaced:

```
Embed
Layer(i, Pre)      residual stream entering layer i            [tokens, hidden]
Layer(i, Attn)     attention probabilities (editable)          [heads, tokens, context]
Layer(i, AttnOut)  what attention adds to the stream           [tokens, hidden]
Layer(i, Mid)      the stream after attention                  [tokens, hidden]
Layer(i, MlpOut)   what the MLP adds                           [tokens, hidden]
Layer(i, Post)     the stream leaving layer i                  [tokens, hidden]
Norm               after the final norm                        [tokens, hidden]
Logits             next-token logits                           [tokens, vocab]
```

- **`Run::step` / `run_to` / `get` / `set`.** Move site by site; whatever you set is what the next stage reads.
- **`Run::finish_with(hook)`.** Run to the end, calling a hook at every site (for steering during generation, for example).
- **`Run::fork`.** Branch from a shared prefix. The KV cache is shared until a branch appends, so scoring N continuations reads the prefix once.
- **`Run::stop`.** End the pass early. `Run::layer_steps` meters the compute actually spent, so the cost of early exit is measured rather than assumed.
- **`Vector`.** A readout tagged with its `Space` (model, site, pooling, template). Comparing vectors from different spaces is an error, not a silent bug.
- **`Choice`.** One of N, with a score, a margin and every alternative: the common answer type of every decider.

## Crates

| crate | contents |
|---|---|
| `latent-core` | `Site`, `Model` (an architecture split into stages), `Run`, `Cache`, `Vector`, `Choice`, the model store |
| `latent-gemma` | Gemma 3 (text) implemented stage by stage on [candle](https://github.com/huggingface/candle), with materialised attention probabilities |
| `latent-algos` | `generate`, `lens`, `readout`, `Bank`, `choose`, `direction` + `steer`, `Tagger` |
| `latent-cli` | the `latent` binary: model store, generation, logit lens |

A new architecture implements `latent_core::Model`: embedding, attention
(returning probabilities and values), attention output, MLP, final norm and
unembedding. Every algorithm then works on it unchanged.

## Algorithms

| | |
|---|---|
| `lens(model, text, k)` | the top-*k* next tokens after each layer (logit lens) |
| `readout(model, text, &Readout)` | one vector from any site, mean- or last-token-pooled, optionally framed by a template. Stops the pass at that site |
| `Bank` | labelled readouts and nearest-example decisions, with optional centring (the residual stream has a few large shared directions that otherwise dominate cosine similarity) |
| `choose(model, head, labels, tail, null_head)` | constrained choice: the log-probability of each label as the continuation, tokenised in context, with the shared prefix forked per label. Optional contextual calibration against a content-free prompt |
| `direction` + `steer` | a direction as the mean difference of two sets of readouts, and a hook that adds it at a site while generating |
| `Tagger` | span extraction: each token takes the type of its nearest example token (read at a chosen layer), and runs of one type become spans |
| `generate` | greedy decoding through a stepped run, with a hook at every site |

## Results on a small model

`cargo run --release --example tour -p latent-algos`, Gemma 3 270M on CPU (a
24-thread AMD Ryzen AI 9 HX 370 laptop).

The decision task: six assistant commands (torch on / off, timer, shopping
list, messaging, stop) with 4–5 example phrasings each, tested on 15 requests
worded unlike the examples.

| | result |
|---|---|
| **decide from the model's own hidden states** (nearest example, centred) | **10/15 from the last token of a framing prompt after layer 6**: 39 % of the forward pass, **40 ms** a decision. Mean pooling also reaches 10/15, after layer 12 |
| constrained choice over the same labels (full pass + one fork per label) | 8/15, 0.76 s raw; 8/15, 1.46 s calibrated |
| logit lens on "What is the capital of France?" | " Paris" becomes the top prediction at **layer 12 of 18** (p = 0.94); layer 11 still prefers "Washington" |
| steering (happy − sad sentences, added after layer 6) | baseline "Mondays are a classic day of rest and reflection." · α = +1: "…a great time to start the week with a sense of purpose and excitement!" · α = −2: "…the worst day of the week, and I don't want to go to work." Strengths around ±4 break fluency |
| extraction (8 example spans, layer 8) | 4/4 sentences exact: duration "ten seconds"; item "bananas"; person "dad" + duration "twenty minutes"; item "rice and sugar". Exact from layer 8 to 14; averaged prototypes instead of nearest tokens get 0–2/4 |

For this model, a decision read from its middle layers is as accurate as or
better than asking it to answer, and about 20× cheaper.

## Getting started

```sh
cargo build --release
./target/release/latent models fetch gemma-3-270m-it    # ~540 MB into the model store
./target/release/latent generate "Write one sentence about Mondays."
./target/release/latent lens "<bos>The capital of France is"
cargo run --release --example tour -p latent-algos      # the results above (~2 min on CPU)
cargo test --release                                     # needs the model; tests skip without it
```

**Model store.** Weights live in `$LATENT_MODELS`, or `/tmp/latent-models` if
that isn't set. `latent models list` shows what's there, and `latent models
clear` deletes it all (it refuses any folder without the store's marker file).

**Devices.** Computation currently runs on CPU in f32 (weights are converted
from bf16 at load).

## Correctness

The tests check that the stepped machinery computes the same thing as a plain
forward pass:
- a prompt fed whole gives the same logits as one fed in two parts through a fork;
- a fork leaves the original run unchanged;
- the sites come in the documented order;
- attention probabilities sum to one;
- editing the stream changes the output;
- an early stop spends exactly the layers it ran, and the stopped run refuses to continue.

## Roadmap

- **Devices and precision:** Metal and CUDA through candle's backends; bf16 / f16 compute; quantised weights.
- **Models:** more Gemma 3 sizes, and other families (Llama, Qwen, Mistral) as `Model` implementations; encoder models such as EmbeddingGemma.
- **Algorithms:** trained linear probes, per-input early exit, task vectors, attention-pattern interventions, KV-cache editing, contrastive decoding between layers, sparse-dictionary features.
- **Batching** of many short readouts in one pass.

## License

MIT
