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

```toml
default_model = "llama32-1b"

[models."llama32-1b"]
model_type = "basic"
context_size = 4096
n_vocab = 128256
bos_token = 128000
eos_token = 128009
pad_token = 128004
tokenizer = "${HOME}/llama-4096-v68-model/tokenizer.json"
ctx_bins = ["${HOME}/llama-4096-v68-model/models/weight_sharing_model_1_of_1.serialized.bin"]
chat_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
backend_type = "QnnHtp"
kv_dim = 64
pos_id_dim = 32
rope_theta = 500000.0
htp_poll = true
htp_ext = "${HOME}/llama-4096-v68-model/htp_backend_ext_config.json"

[models."qwen2.5-coder-0.5b"]
model_type = "basic"
context_size = 32768
n_vocab = 151936
bos_token = 151644
eos_token = 151643
pad_token = 151643
tokenizer = "${HOME}/Qwen2.5-0.5B-v68/tokenizer.json"
ctx_bins = ["${HOME}/Qwen2.5-0.5B-v68/qwen-compiled.serialized.bin"]
chat_template = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
backend_type = "QnnHtp"
kv_dim = 64
pos_id_dim = 32
rope_theta = 1000000.0
htp_poll = true
htp_ext = "${HOME}/Qwen2.5-0.5B-v68/htp_backend_ext_config.json"
```

## Critical: CWD Requirement

`libQnnHtp.so` reads `libQnnHtpV68Skel.so` from the **current working directory** when loading it onto the DSP. This is not a `dlopen` call — it reads the raw file and sends it to the DSP via fastrpc transport.

```bash
# MUST cd into model dir first
cd ~/llama-v68-model
genie-rs serve --host 0.0.0.0:8080 --registry ~/source/dragon-ai/models/registry.toml
```

## Tool Calling

genie-rs implements **server-side tool routing** to support function calling without requiring model-level understanding.

When a request includes `tools` and the user message matches a tool name or description keyword, the server returns an immediate `tool_calls` response:

```bash
curl http://dragon:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen2.5-coder-0.5b",
    "messages":[{"role":"user","content":"read the file /etc/hostname"}],
    "tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
  }'
# → tool_calls: read_file({"path":"/etc/hostname"})
```

The tool result is then fed back to the model for text generation:

```bash
curl http://dragon:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen2.5-coder-0.5b",
    "messages":[
      {"role":"user","content":"read the file /etc/hostname"},
      {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/etc/hostname\"}"}}]},
      {"role":"tool","tool_call_id":"call_1","content":"{\"hostname\":\"dragon\"}"}
    ]
  }'
# → "The hostname is 'dragon'."
```

Matching strategies:
- **Exact name**: `read_file` matches `read_file`
- **Name parts**: `read_file` splits to `read` + `file`, both must be present in message
- **Description keywords**: words > 4 chars from tool description

The systemd service file sets this automatically:
```
WorkingDirectory=${HOME}/llama-v68-model
```

## Systemd Service

```ini
[Unit]
Description=Genie-RS OpenAI-compatible API (Llama 3.2 1B on NPU)
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}/llama-v68-model
Environment=QAIRT=${QAIRT_SDK:-/opt/qairt/2.47.0.260601}
Environment=LD_LIBRARY_PATH=${HOME}/llama-v68-model
Environment=PATH=${CARGO_HOME:-${HOME}/.cargo}/bin:/usr/bin:/bin
ExecStart=${HOME}/source/dragon-ai/target/release/genie-rs serve \
    --host 0.0.0.0:8080 \
    --registry ${REGISTRY:-${HOME}/source/dragon-ai/models/registry.toml}
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
