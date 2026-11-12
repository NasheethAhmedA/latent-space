//! Model weights live only in a temporary folder: `$LATENT_MODELS`, else
//! `/tmp/latent-models` (RAM-backed here, gone at reboot), and `clear()`
//! deletes it in one step. It refuses any folder without its marker file.

use std::path::PathBuf;

pub const DEFAULT_DIR: &str = "/tmp/latent-models";
const MARKER: &str = ".latent-models";

pub fn models_dir() -> PathBuf {
    std::env::var_os("LATENT_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR))
}

pub fn model_dir(id: &str) -> PathBuf {
    models_dir().join(id)
}

pub fn ensure() -> std::io::Result<PathBuf> {
    let d = models_dir();
    std::fs::create_dir_all(&d)?;
    let m = d.join(MARKER);
    if !m.exists() {
        std::fs::write(
            &m,
            "latent model store. Safe to delete entirely: `latent models clear`.\n",
        )?;
    }
    Ok(d)
}

/// Bytes under a path.
pub fn size_of(p: &std::path::Path) -> u64 {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => std::fs::read_dir(p)
            .map(|rd| rd.flatten().map(|e| size_of(&e.path())).sum())
            .unwrap_or(0),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

/// Delete the whole store; returns the bytes freed.
pub fn clear() -> std::io::Result<u64> {
    let d = models_dir();
    if !d.exists() {
        return Ok(0);
    }
    if !d.join(MARKER).exists() {
        return Err(std::io::Error::other(format!(
            "{} has no {MARKER} marker: not a latent model store, refusing to delete it",
            d.display()
        )));
    }
    let n = size_of(&d);
    std::fs::remove_dir_all(&d)?;
    Ok(n)
}
