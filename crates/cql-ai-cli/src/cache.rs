//! Model resolution: explicit directory, `~/.cache/cql-ai`, or download from Hugging Face.

use anyhow::{Context, Result, bail};
use cql_ai_model::io::{CONFIG_FILE, WEIGHTS_FILE};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Hugging Face repository holding the published weights.
pub const HF_REPO: &str = "Max-imumlikelihood/cql-anti-pattern-transformer";

/// `$CQL_AI_HF_REPO` overrides [`HF_REPO`] (forks, a differently named Hugging Face account).
fn repo() -> String {
    std::env::var("CQL_AI_HF_REPO").unwrap_or_else(|_| HF_REPO.to_string())
}

/// `$CQL_AI_CACHE`, else `$XDG_CACHE_HOME/cql-ai`, else `~/.cache/cql-ai`.
pub fn cache_root() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("CQL_AI_CACHE") {
        return Ok(p.into());
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(Path::new(&p).join("cql-ai"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("cannot locate a home directory; set CQL_AI_CACHE")?;
    Ok(Path::new(&home).join(".cache").join("cql-ai"))
}

fn complete(dir: &Path) -> bool {
    dir.join(WEIGHTS_FILE).is_file() && dir.join(CONFIG_FILE).is_file()
}

/// Returns a directory containing `model.safetensors` and `config.json`.
///
/// Order: `explicit` → cache hit → download (`HF_ENDPOINT`/`CQL_AI_HF_BASE` override the host).
pub fn resolve_model(explicit: Option<&Path>, refresh: bool, offline: bool) -> Result<PathBuf> {
    if let Some(dir) = explicit {
        if !complete(dir) {
            bail!(
                "{} must contain {WEIGHTS_FILE} and {CONFIG_FILE}",
                dir.display()
            );
        }
        return Ok(dir.to_path_buf());
    }
    let dir = cache_root()?.join(repo().replace('/', "--"));
    if complete(&dir) && !refresh {
        return Ok(dir);
    }
    if offline {
        bail!(
            "model not cached in {} and --offline was given",
            dir.display()
        );
    }
    std::fs::create_dir_all(&dir)?;
    let base = std::env::var("CQL_AI_HF_BASE")
        .or_else(|_| std::env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| "https://huggingface.co".into());
    for file in [CONFIG_FILE, WEIGHTS_FILE] {
        let url = format!(
            "{}/{}/resolve/main/{file}",
            base.trim_end_matches('/'),
            repo()
        );
        download(&url, &dir.join(file)).with_context(|| {
            format!("downloading {url}\nhint: pass --model <dir> with a local model.safetensors + config.json")
        })?;
    }
    Ok(dir)
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let mut req = agent.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let resp = req.call()?;
    let tmp = dest.with_extension("part");
    let mut out = std::fs::File::create(&tmp)?;
    std::io::copy(&mut resp.into_body().into_reader(), &mut out)?;
    drop(out);
    std::fs::rename(tmp, dest)?;
    Ok(())
}
