# HTP Backend Configuration

## Config Generation

The genie-rs server generates the QNN dialog config JSON at runtime based on fields in `registry.toml`. This document explains the generated config structure.

## Generated Config (Llama 3.2 1B)

When genie-rs loads the `llama32-1b` model (backend_type `QnnHtp`), it produces this config:

```json
{
  "dialog": {
    "version": 1,
    "type": "basic",
    "max-num-tokens": 512,
    "context": {
      "version": 1,
      "size": 1024,
      "n-vocab": 128256,
      "bos-token": 128000,
      "eos-token": 128009,
      "pad-token": 128004
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
      "path": "/home/daniel/llama-v68-model/tokenizer.json"
    },
    "engine": {
      "version": 1,
      "n-threads": 3,
      "backend": {
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "use-mmap": true,
          "spill-fill-bufsize": 0,
          "mmap-budget": 0,
          "poll": true,
          "cpu-mask": "0xe0",
          "kv-dim": 64,
          "allow-async-init": true
        },
        "extensions": "/home/daniel/llama-v68-model/htp_backend_ext_config.json"
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "ctx-bins": [
            "/home/daniel/llama-v68-model/models/weight_sharing_model_1_of_1.serialized.bin"
          ]
        },
        "positional-encoding": {
          "type": "rope",
          "rope-dim": 32,
          "rope-theta": 500000,
          "rope-scaling": {
            "rope-type": "llama3",
            "factor": 32.0,
            "low-freq-factor": 1.0,
            "high-freq-factor": 4.0,
            "original-max-position-embeddings": 8192
          }
        }
      }
    }
  }
}
```

## HTP Backend Extensions Config

File: `htp_backend_ext_config.json`

```json
{
  "devices": [
    {
      "soc_id": 35,
      "dsp_arch": "v68",
      "cores": [
        {
          "core_id": 0,
          "perf_profile": "burst",
          "rpc_control_latency": 100
        }
      ]
    }
  ],
  "memory": {
    "mem_type": "shared_buffer"
  }
}
```

| Field | Value | Meaning |
|-------|-------|---------|
| `soc_id` | 35 | QCS6490 SoC identifier |
| `dsp_arch` | `v68` | Hexagon DSP architecture version |
| `core_id` | 0 | Use compute core 0 (only one v68 core on QCS6490) |
| `perf_profile` | `burst` | Maximum performance profile |
| `rpc_control_latency` | 100 | RPC latency in microseconds |
| `mem_type` | `shared_buffer` | Shared buffer memory mode (uses DMA pool) |

## Llama 3 RoPE Scaling

The `kv-dim: 64` setting triggers Llama 3 rope scaling in `qnn_config.rs`:

```rust
if kv_dim == 64 {
    pos_enc["rope-scaling"] = json!({
        "rope-type": "llama3",
        "factor": 32.0,
        "low-freq-factor": 1.0,
        "high-freq-factor": 4.0,
        "original-max-position-embeddings": 8192
    });
}
```

This is required for Llama 3.x models. The pre-compiled context binary (`weight_sharing_model_1_of_1.serialized.bin`) includes the model weights quantized to INT8 and the attention operation is configured for Grouped Query Attention (GQA).

## Token IDs (Llama 3)

| Token | ID | Purpose |
|-------|-----|---------|
| BOS | 128000 | `<|begin_of_text|>` |
| EOS | 128009 | `<|eot_id|>` (end of turn) |
| PAD | 128004 | `<|reserved_special_token_0|>` (padding) |

## Chat Template

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```
