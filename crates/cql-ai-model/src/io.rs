//! Persistence: `model.safetensors` + `config.json`.

use crate::model::{AntiPatternNet, ModelConfig};
use anyhow::{Context, Result};
use burn::prelude::*;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use std::path::Path;

pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";

/// Writes weights and config into `dir`.
pub fn save(model: &AntiPatternNet, config: &ModelConfig, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    config
        .save(dir.join(CONFIG_FILE))
        .context("writing config.json")?;
    let mut store = SafetensorsStore::from_file(dir.join(WEIGHTS_FILE))
        .metadata("format", "cql-ai/anti-pattern-transformer")
        .overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|e| anyhow::anyhow!("saving safetensors: {e:?}"))?;
    Ok(())
}

/// Loads a model saved by [`save`] onto `device`.
pub fn load(dir: &Path, device: &Device) -> Result<(AntiPatternNet, ModelConfig)> {
    let config = ModelConfig::load(dir.join(CONFIG_FILE)).context("reading config.json")?;
    let mut model = config.init(device);
    let mut store = SafetensorsStore::from_file(dir.join(WEIGHTS_FILE));
    let result = model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("loading safetensors: {e:?}"))?;
    if !result.errors.is_empty() || !result.missing.is_empty() {
        anyhow::bail!("weights do not match the model: {result}");
    }
    Ok((model, config))
}
