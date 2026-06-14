# genie-rs API Server

The API server is a Rust binary that wraps Qualcomm's Genie C SDK for ARM64 Linux and exposes an OpenAI-compatible HTTP API.

## Source Repository

The genie-rs project lives at `~/source/dragon-ai/` on the Dragon board (separate from this repo). This section documents the changes needed to configure it for NPU inference.

## Architecture

```
                    ┌─────────────────────────────────────┐
HTTP request ───────┤  axum Router                        │
                    │  /v1/chat/completions                │
                    │  /v1/models                          │
                    │  /v1/admin/stop                      │
                    │  /v1/admin/clear                     │
                    ├─────────────────────────────────────┤
                    │  build_prompt()                      │
                    │    → chat_template with system+user  │
                    ├─────────────────────────────────────┤
                    │  GenieContext::run_query()            │
                    │    → GenieDialog_query() with callback│
                    ├─────────────────────────────────────┤
                    │  Genie C SDK (libGenie.so)           │
                    │  QNN HTP backend (libQnnHtp.so)      │
                    │  fastrpc → Hexagon v68 NPU          │
                    └─────────────────────────────────────┘
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI arg parsing (`serve`, `list`), registry loading, working-dir support |
| `src/config/registry.rs` | `ModelRegistry` + `ModelEntry` deserialization from `registry.toml` |
| `src/config/qnn_config.rs` | Generates the JSON config consumed by `GenieDialogConfig_createFromJson` |
| `src/context/genie_context.rs` | Wrapper around GenieDialog C API (create, query, tokenize, reset) |
| `src/server/mod.rs` | axum Router setup with routes |
| `src/server/routes.rs` | OpenAI-compatible HTTP handlers (chat completions, streaming, models list) |
| `build.rs` | bindgen for Genie C headers, rpath for QAIRT libs |
| `models/registry.toml` | Model registry — defines available models and their config overrides |

## Patches Applied

### 1. registry.rs — New fields for HTP models

Added to `ModelEntry`:
```rust
pub pad_token: Option<u32>,
pub kv_dim: Option<u32>,
pub pos_id_dim: Option<u32>,
pub rope_theta: Option<f64>,
pub htp_poll: Option<bool>,
#[serde(default = "default_backend")]
pub backend_type: String,   // "QnnHtp" (default) or "QnnGenAiTransformer"
```

### 2. qnn_config.rs — HTP backend config

Added:
- `build_htp_backend()` — generates the `QnnHtp` backend JSON section
- `build_genai_backend()` — generates the `QnnGenAiTransformer` backend section for CPU fallback
- Positional encoding (RoPE) with Llama 3 scaling detection (`kv_dim == 64 → llama3 rope-scaling`)
- Pad token (`pad_token` or falls back to `eos_token`)

### 3. build.rs — QAIRT 2.47 paths + rpath

Updated:
- QAIRT SDK path to `2.47.0.260601`
- Library dir: `lib/aarch64-oe-linux-gcc11.2`
- Added rpath entries for model dirs (so `libQnnHtp.so` can find its dependencies at runtime)

## Model Registry (`registry.toml`)

```toml
default_model = "llama32-1b"

[models."llama32-1b"]
model_type = "basic"
context_size = 1024
n_vocab = 128256
bos_token = 128000
eos_token = 128009
pad_token = 128004
tokenizer = "/home/daniel/llama-v68-model/tokenizer.json"
ctx_bins = ["/home/daniel/llama-v68-model/models/weight_sharing_model_1_of_1.serialized.bin"]
chat_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
backend_type = "QnnHtp"
kv_dim = 64
pos_id_dim = 32
rope_theta = 500000.0
htp_poll = true
htp_ext = "/home/daniel/llama-v68-model/htp_backend_ext_config.json"
```

## Critical: CWD Requirement

`libQnnHtp.so` reads `libQnnHtpV68Skel.so` from the **current working directory** when loading it onto the DSP. This is not a `dlopen` call — it reads the raw file and sends it to the DSP via fastrpc transport.

```bash
# MUST cd into model dir first
cd ~/llama-v68-model
genie-rs serve --host 0.0.0.0:8080 --registry ~/source/dragon-ai/models/registry.toml
```

The systemd service file sets this automatically:
```
WorkingDirectory=/home/daniel/llama-v68-model
```

## Systemd Service

```ini
[Unit]
Description=Genie-RS OpenAI-compatible API (Llama 3.2 1B on NPU)
After=network.target

[Service]
Type=simple
User=daniel
WorkingDirectory=/home/daniel/llama-v68-model
Environment=QAIRT=/home/daniel/qairt/2.47.0.260601
Environment=LD_LIBRARY_PATH=/home/daniel/llama-v68-model
Environment=PATH=/home/daniel/.cargo/bin:/usr/bin:/bin
ExecStart=/home/daniel/source/dragon-ai/target/release/genie-rs serve \
    --host 0.0.0.0:8080 \
    --registry /home/daniel/source/dragon-ai/models/registry.toml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Building

```bash
# On the Dragon board:
cd ~/source/dragon-ai
source ~/.cargo/env
export QAIRT=$HOME/qairt/2.47.0.260601
cargo build --release
```

The binary at `target/release/genie-rs` is statically linked against Rust std but dynamically links Genie/QNN libraries at runtime.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (non-streaming) |
| POST | `/v1/admin/stop` | Stop current generation |
| POST | `/v1/admin/clear` | Reset dialog context |
