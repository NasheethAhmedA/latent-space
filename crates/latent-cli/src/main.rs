//! `latent`: the model store, and quick looks inside a model.
use latent_core::store;

fn usage() -> ! {
    eprintln!("usage:\n  latent models [path | list | clear | fetch ID]   the temp model store ($LATENT_MODELS, else /tmp/latent-models)\n  latent generate \"PROMPT\" [--max N] [--model ID]   greedy reply through a stepped run\n  latent lens \"TEXT\" [--model ID]                  what each layer would say next (logit lens)");
    std::process::exit(2)
}

fn main() {
    if let Err(e) = real() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn real() -> latent_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg(&args, "--model").unwrap_or("gemma-3-270m-it".into());
    match args.first().map(String::as_str) {
        Some("models") => match args.get(1).map(String::as_str) {
            Some("path") | None => println!("{}", store::models_dir().display()),
            Some("list") => {
                for e in std::fs::read_dir(store::models_dir())
                    .into_iter()
                    .flatten()
                    .flatten()
                {
                    if e.path().is_dir() {
                        println!(
                            "{:<28} {:>6} MB",
                            e.file_name().to_string_lossy(),
                            store::size_of(&e.path()) / 1_000_000
                        );
                    }
                }
            }
            Some("clear") => println!(
                "deleted {} MB",
                store::clear().map_err(|e| latent_core::Error::Msg(e.to_string()))? / 1_000_000
            ),
            Some("fetch") => fetch(args.get(2).map(String::as_str).unwrap_or("gemma-3-270m-it"))?,
            _ => usage(),
        },
        Some("generate") => {
            let m = latent_gemma::load(&model_id)?;
            let max: usize = arg(&args, "--max")
                .and_then(|s| s.parse().ok())
                .unwrap_or(48);
            let t0 = std::time::Instant::now();
            let out = latent_algos::generate(
                &m,
                &latent_gemma::chat(args.get(1).map(String::as_str).unwrap_or_else(|| usage())),
                max,
                |_, _| Ok(()),
            )?;
            println!("{out}\n({:?})", t0.elapsed());
        }
        Some("lens") => {
            let m = latent_gemma::load(&model_id)?;
            let text = args.get(1).cloned().unwrap_or_else(|| usage());
            for (layer, top) in latent_algos::lens(&m, &text, 3)? {
                println!(
                    "{:>8}  {}",
                    layer,
                    top.iter()
                        .map(|(t, p)| format!("{:?} {:.2}", t, p))
                        .collect::<Vec<_>>()
                        .join("  ")
                );
            }
        }
        _ => usage(),
    }
    Ok(())
}

fn fetch(id: &str) -> latent_core::Result<()> {
    let repo = match id {
        "gemma-3-270m-it" => "unsloth/gemma-3-270m-it",
        other => {
            return latent_core::err(format!("unknown model `{other}` (known: gemma-3-270m-it)"))
        }
    };
    store::ensure().map_err(|e| latent_core::Error::Msg(e.to_string()))?;
    let d = store::model_dir(id);
    std::fs::create_dir_all(&d).map_err(|e| latent_core::Error::Msg(e.to_string()))?;
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        if d.join(f).exists() {
            continue;
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{f}");
        println!("fetching {url}");
        let st = std::process::Command::new("curl")
            .args(["-sfL", "-o"])
            .arg(d.join(f))
            .arg(&url)
            .status()
            .map_err(|e| latent_core::Error::Msg(e.to_string()))?;
        if !st.success() {
            return latent_core::err(format!("download failed: {url}"));
        }
    }
    Ok(())
}
