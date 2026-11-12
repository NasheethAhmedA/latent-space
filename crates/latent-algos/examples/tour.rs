//! A tour of latent space programming on one small model (Gemma 3 270M):
//!
//! 1. Decide among six assistant commands from the stream at each layer (no
//!    embedding model: the LLM's own vectors), with two readouts, and what
//!    fraction of a forward pass each decision costs.
//! 2. The same decision by constrained choice (logits), raw and calibrated.
//! 3. Steering: a direction from two sets of sentences, added while generating.
//! 4. Extraction: slot spans from per-token vectors and a few examples.
use latent_algos::{choose, direction, generate, steer, Bank, Readout, Tagger};
use latent_core::{Model, Pool, Site};

const COMMANDS: &[(&str, &[&str])] = &[
    (
        "torch_on",
        &[
            "turn on the torch",
            "torch on",
            "switch on the flashlight",
            "light please",
            "turn the light on",
        ],
    ),
    (
        "torch_off",
        &[
            "turn off the torch",
            "torch off",
            "switch off the flashlight",
            "turn the light off",
        ],
    ),
    (
        "set_timer",
        &[
            "set a timer",
            "set a timer for five minutes",
            "start a timer",
            "timer for 10 minutes",
        ],
    ),
    (
        "shopping",
        &[
            "add to my shopping list",
            "add milk to my shopping list",
            "put eggs on the shopping list",
            "add something to the list",
        ],
    ),
    (
        "send_message",
        &[
            "message mom",
            "text mom that i'm running late",
            "tell mom that i'll be home soon",
            "send mom a message",
        ],
    ),
    (
        "stop_listening",
        &[
            "stop listening",
            "go to sleep",
            "turn yourself off",
            "stop listening for now",
        ],
    ),
];
const TESTS: &[(&str, &str)] = &[
    ("it's too dark in here, give me some light", "torch_on"),
    ("kill the flashlight", "torch_off"),
    ("I can see now, the light can go off", "torch_off"),
    ("remind me in twenty minutes", "set_timer"),
    ("countdown three minutes for the eggs", "set_timer"),
    ("we're out of bread", "shopping"),
    ("don't let me forget to buy apples", "shopping"),
    ("let my mother know I'm stuck in traffic", "send_message"),
    ("tell mum dinner was great", "send_message"),
    ("that's enough for today, rest", "stop_listening"),
    ("please be quiet and stop hearing me", "stop_listening"),
    ("flashlight", "torch_on"),
    ("hey, turn on the torch.", "torch_on"),
    ("please turn off the torch now.", "torch_off"),
    ("set an alarm for ten minutes", "set_timer"),
];

fn main() -> latent_core::Result<()> {
    let m = latent_gemma::load("gemma-3-270m-it")?;
    let layers = m.spec().layers;
    let examples: Vec<(&str, Vec<&str>)> = COMMANDS.iter().map(|(l, e)| (*l, e.to_vec())).collect();
    let template = "<bos><start_of_turn>user\nWhat does this request ask for: \"{}\"<end_of_turn>\n<start_of_turn>model\nIt asks to";

    println!(
        "== 1 · deciding from the stream at layer k (nearest example, centred), {} unseen requests",
        TESTS.len()
    );
    println!(
        "{:<34} {}",
        "readout",
        (0..layers)
            .step_by(1)
            .map(|l| format!("{:>4}", format!("L{l}")))
            .collect::<String>()
    );
    let mut best = (0usize, String::new(), 0usize);
    for (name, r) in [
        (
            "mean of the request's tokens",
            Readout::new(Site::Embed, Pool::Mean),
        ),
        (
            "last token of a framing prompt",
            Readout::new(Site::Embed, Pool::Last).framed(template),
        ),
    ] {
        let mut row = String::new();
        for l in 0..layers {
            let mut r = r.clone();
            r.site = Site::post(l);
            let bank = Bank::build(&m, r, &examples, true)?;
            let mut right = 0;
            for (t, want) in TESTS {
                if bank.decide(&m, t)?.0.label == *want {
                    right += 1;
                }
            }
            row += &format!("{right:>4}");
            if right > best.0 || (right == best.0 && l < best.2) {
                best = (right, name.to_string(), l);
            }
        }
        println!("{name:<34}{row}");
    }
    let (right, how, l) = &best;
    let mut r = if how.starts_with("mean") {
        Readout::new(Site::post(*l), Pool::Mean)
    } else {
        Readout::new(Site::post(*l), Pool::Last).framed(template)
    };
    r.site = Site::post(*l);
    let bank = Bank::build(&m, r, &examples, true)?;
    let t0 = std::time::Instant::now();
    let mut steps = 0;
    for (t, _) in TESTS {
        steps += bank.decide(&m, t)?.1;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / TESTS.len() as f64;
    println!("best: {right}/{} with the {how} after layer {l}: {:.0} % of the layers, {ms:.0} ms a decision ({} layer-token steps each)",
        TESTS.len(), 100.0 * (*l as f64 + 1.0) / layers as f64, steps / TESTS.len());

    println!(
        "\n== 2 · constrained choice from the logits (the full pass, then one fork per label)"
    );
    let labels: Vec<String> = COMMANDS.iter().map(|c| c.0.to_string()).collect();
    let listing: String = COMMANDS
        .iter()
        .map(|(l, e)| format!("- {l}: e.g. \"{}\"\n", e[..2].join("\", \"")))
        .collect();
    let head = |req: &str| {
        latent_gemma::chat(&format!("Pick the option that the request is asking for.\n\nOptions:\n{listing}\nRequest: \"{req}\"\nAnswer with the option name only.")) + "- "
    };
    for calibrated in [false, true] {
        let t0 = std::time::Instant::now();
        let mut right = 0;
        for (t, want) in TESTS {
            let c = choose(
                &m,
                &head(t),
                &labels,
                "",
                calibrated.then(|| head("N/A")).as_deref(),
            )?;
            if c.label == *want {
                right += 1;
            }
        }
        println!(
            "{:<12} {right}/{} · {:.0} ms a decision",
            if calibrated { "calibrated" } else { "raw" },
            TESTS.len(),
            t0.elapsed().as_secs_f64() * 1000.0 / TESTS.len() as f64
        );
    }

    println!("\n== 3 · steering: a direction from two sets of sentences, added after layer 6");
    let site = Site::post(6);
    let dir = direction(
        &m,
        site,
        &[
            "I am so happy today!",
            "This is wonderful news.",
            "I love this, it's fantastic.",
            "What a beautiful, joyful day.",
        ],
        &[
            "I am so sad today.",
            "This is terrible news.",
            "I hate this, it's awful.",
            "What a gloomy, miserable day.",
        ],
    )?;
    let prompt = latent_gemma::chat("Write one sentence about Mondays.");
    for alpha in [0.0, 1.0, -2.0] {
        println!(
            "α = {alpha:>4}: {}",
            generate(&m, &prompt, 30, steer(site, dir.clone(), alpha))?.replace('\n', " ")
        );
    }

    println!("\n== 4 · extraction: each token takes the type of its nearest example token (8 example spans), read after layer 8");
    let tagger = Tagger::build(
        &m,
        Site::post(8),
        &[
            ("duration", "set a timer for five minutes", "five minutes"),
            ("duration", "remind me in an hour", "an hour"),
            ("duration", "wake me up in 20 minutes", "20 minutes"),
            ("item", "add milk to my shopping list", "milk"),
            ("item", "put eggs and bread on the list", "eggs and bread"),
            ("item", "we need more coffee", "coffee"),
            ("person", "text mom that i'm late", "mom"),
            ("person", "call my brother Sam", "my brother Sam"),
        ],
    )?;
    for t in [
        "start a countdown of ten seconds",
        "add bananas to the list",
        "tell dad I'll be home in twenty minutes",
        "we are out of rice and sugar",
    ] {
        let spans = tagger.tag(&m, t)?;
        println!(
            "“{t}” → {}",
            if spans.is_empty() {
                "nothing".into()
            } else {
                spans
                    .iter()
                    .map(|(k, s, c)| format!("{k}: “{s}” ({c:.2})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
    }
    Ok(())
}
