use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub default_model: String,
    pub models: HashMap<String, ModelEntry>,
}
impl ModelRegistry {
    /// Expand environment variables in all model entry paths.
    pub fn expand_all_paths(&mut self) {
        for entry in self.models.values_mut() {
            entry.expand_paths();
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub model_type: String,
    pub context_size: u32,
    pub n_vocab: u32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: Option<u32>,
    #[serde(default)]
    pub htp_ext: String,
    pub tokenizer: String,
    pub ctx_bins: Vec<String>,
    pub chat_template: String,
    pub kv_dim: Option<u32>,
    pub pos_id_dim: Option<u32>,
    pub rope_theta: Option<f64>,
    pub htp_poll: Option<bool>,
    #[serde(default = "default_backend")]
    pub backend_type: String,
    pub ssd: Option<SsdConfig>,
}
impl ModelEntry {
    /// Expand environment variables in all path fields.
    /// Supports ${VAR} and ${VAR:-default} syntax.
    pub fn expand_paths(&mut self) {
        let expand = |s: &mut String| {
            let mut result = String::new();
            let mut rest = s.as_str();
            while let Some(start) = rest.find("${") {
                result.push_str(&rest[..start]);
                let inner = &rest[start+2..];
                let end = inner.find('}').unwrap_or(inner.len());
                let var_expr = &inner[..end];
                let (var, default) = match var_expr.split_once(":-") {
                    Some((v, d)) => (v, Some(d)),
                    None => (var_expr, None),
                };
                let val = std::env::var(var).ok()
                    .or_else(|| default.map(|d| d.to_string()))
                    .unwrap_or_default();
                result.push_str(&val);
                rest = &inner[end+1..];
            }
            result.push_str(rest);
            *s = result;
        };
        expand(&mut self.tokenizer);
        expand(&mut self.chat_template);
        expand(&mut self.htp_ext);
        for bin in &mut self.ctx_bins {
            expand(bin);
        }
    }
}

fn default_backend() -> String {
    "QnnHtp".to_string()
}

#[derive(Debug, Deserialize)]
pub struct SsdConfig {
    pub forecast_token_count: u32,
    pub forecast_prefix: u32,
    pub forecast_prefix_name: String,
    pub branches: Vec<u32>,
    pub n_streams: u32,
    pub p_threshold: f32,
}
