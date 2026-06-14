use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub default_model: String,
    pub models: HashMap<String, ModelEntry>,
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
