use super::registry::{ModelEntry, SsdConfig};
use serde_json::{json, Value};

pub fn generate_qnn_config(model: &ModelEntry) -> String {
    let kv_dim = model.kv_dim.unwrap_or(128);
    let pos_id_dim = model.pos_id_dim.unwrap_or(64);
    let max_tokens = model.context_size / 2;
    let backend = match model.backend_type.as_str() {
        "QnnGenAiTransformer" => build_genai_backend(),
        _ => build_htp_backend(model, kv_dim, pos_id_dim),
    };
    let mut model_cfg = match model.backend_type.as_str() {
        "QnnGenAiTransformer" => json!({
            "version": 1,
            "type": "library",
            "library": {
                "version": 1,
                "model-bin": model.ctx_bins.first().map(|s| s.as_str()).unwrap_or("")
            }
        }),
        _ => json!({
            "version": 1,
            "type": "binary",
            "binary": {
                "version": 1,
                "ctx-bins": model.ctx_bins
            }
        }),
    };
    // Add positional-encoding for HTP models with rope
    if model.backend_type != "QnnGenAiTransformer" && model.pos_id_dim.is_some() {
        let rope_theta = model.rope_theta.unwrap_or(1000000.0);
        let mut pos_enc = json!({
            "type": "rope",
            "rope-dim": pos_id_dim,
            "rope-theta": rope_theta
        });
        // Add Llama 3 rope scaling for models with kv_dim=64 (like Llama 3.2 1B)
        if kv_dim == 64 {
            pos_enc["rope-scaling"] = json!({
                "rope-type": "llama3",
                "factor": 32.0,
                "low-freq-factor": 1.0,
                "high-freq-factor": 4.0,
                "original-max-position-embeddings": 8192
            });
        }
        model_cfg["positional-encoding"] = pos_enc;
    }
    let pad_token = model.pad_token.unwrap_or(model.eos_token);
    let mut dialog = json!({
        "version": 1,
        "type": model.model_type,
        "max-num-tokens": max_tokens,
        "context": {
            "version": 1,
            "size": model.context_size,
            "n-vocab": model.n_vocab,
            "bos-token": model.bos_token,
            "eos-token": model.eos_token,
            "pad-token": pad_token
        },
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.8,
            "top-k": 1,
            "top-p": 0.95
        },
        "tokenizer": {
            "version": 1,
            "path": model.tokenizer
        },
        "engine": {
            "version": 1,
            "n-threads": 3,
            "backend": backend,
            "model": model_cfg
        }
    });
    if let Some(ref ssd) = model.ssd {
        dialog["ssd-q1"] = build_ssd_config(ssd);
    }
    let config = json!({ "dialog": dialog });
    serde_json::to_string(&config).unwrap()
}

fn build_htp_backend(model: &ModelEntry, kv_dim: u32, pos_id_dim: u32) -> Value {
    let poll = model.htp_poll.unwrap_or(false);
    let mut htp = json!({
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
            "version": 1,
            "use-mmap": true,
            "spill-fill-bufsize": 0,
            "mmap-budget": 0,
            "poll": poll,
            "cpu-mask": "0xe0",
            "kv-dim": kv_dim,
            "allow-async-init": true
        }
    });
    if !model.htp_ext.is_empty() {
        htp["extensions"] = json!(model.htp_ext);
    }
    htp
}

fn build_genai_backend() -> Value {
    json!({
        "version": 1,
        "type": "QnnGenAiTransformer",
        "QnnGenAiTransformer": {
            "version": 1,
            "kv-quantization": false
        }
    })
}

fn build_ssd_config(ssd: &SsdConfig) -> Value {
    json!({
        "version": 1,
        "ssd-version": 1,
        "forecast-token-count": ssd.forecast_token_count,
        "forecast-prefix": ssd.forecast_prefix,
        "forecast-prefix-name": ssd.forecast_prefix_name,
        "branches": ssd.branches,
        "n-streams": ssd.n_streams,
        "p-threshold": ssd.p_threshold
    })
}
