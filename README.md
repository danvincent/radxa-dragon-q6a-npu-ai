# Dragon NPU API — LLM Inference on Hexagon v68 NPU

Run OpenAI-compatible chat completions on the **Radxa Dragon Q6A** board's **Qualcomm Hexagon v68 NPU** (QCS6490).

## Models

| Model | Backend | Context | Quant | Speed | Source |
|-------|---------|---------|-------|-------|--------|
| `llama32-1b` | HTP V68 (context binary) | 4096 | INT8 | ~6-8 tok/s | ModelScope |
| `qwen2.5-coder-0.5b` | HTP V68 (context binary) | 32768 | INT8 | ~6-8 tok/s | Pre-compiled |
| `qwen2.5-coder-1.5b` | HTP V68 (context binary) | 32768 | INT8 | ~6-8 tok/s | Custom build |

## Features

- **OpenAI-compatible API**: `/v1/chat/completions` — drop-in for any OpenAI client
- **Tool calling**: server-side routing detects tool intent from user message; returns proper `tool_calls` response
- **Streaming**: SSE chunked streaming for real-time text generation
- **Opencode integration**: `opencode run -m dragon-npu/llama32-1b` works out of the box

## Repo Contents

| Path | Description |
|------|-------------|
| `genie-rs/` | **genie-rs Rust source** — the API server |
| `genie-rs/src/server/routes.rs` | Chat completions handler + tool routing logic |
| `genie-rs/src/config/` | Registry parsing + GenieDialog config generation |
| `genie-rs/src/context/genie_context.rs` | GenieDialog C API wrapper |
| `docs/07-model-build-pipeline.md` | ONNX → HTP context binary build pipeline |
| `genie-rs/models/registry.toml` | Model registry — Llama + Qwen entries |
| `docs/01-overview.md` | Architecture walkthrough |
| `docs/02-npu-dma-fix.md` | DTB patching for fastrpc DMA |
| `docs/03-genie-rs-api.md` | API server setup |
| `docs/04-htp-config.md` | HTP backend configuration |
| `docs/05-kernel-update-procedure.md` | Post-kernel-update steps |
| `docs/06-troubleshooting.md` | Common issues |
| `npu-dma-fix/` | DTBO overlay source + build scripts |
| `genie-configs/` | Reference HTP config files |
| `scripts/genie-rs.service` | Systemd unit |
| `scripts/start-genie-server.sh` | Manual startup script |
| `scripts/deploy-to-dragon.sh` | Deploy repo to Dragon |

## Quick Start

### Prerequisites

- Radxa Dragon Q6A with Ubuntu 24.04, kernel 6.18.2-4-qcom
- QAIRT 2.47 SDK at `~/qairt/2.47.0.260601/`
- Rust toolchain (on Dragon)
- Model files at `~/llama-4096-v68-model/` (Llama) and `~/Qwen2.5-0.5B-v68/` (Qwen)
```json
{
  "providerId": "dragon-npu",
  "type": "openai",
  "apiBase": "http://dragon:8080/v1",
  "models": {
    "llama32-1b": { "tool_call": true, "maxOutput": 512 },
    "qwen2.5-coder-0.5b": { "tool_call": true, "maxOutput": 2048 },
    "qwen2.5-coder-1.5b": { "tool_call": true, "maxOutput": 4096 }
  }
}
```
```

### 2. Apply NPU DMA fix

```bash
sudo apt install device-tree-compiler
dtc -@ -I dts -O dtb -o npu-dma-fix.dtbo npu-dma-fix-overlay.dts
ESP="/boot/efi/RadxaOS/6.18.2-4-qcom"
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb" /tmp/orig.dtb
fdtoverlay -i /tmp/orig.dtb -o /tmp/patched.dtb npu-dma-fix.dtbo
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb"{,.stock}
sudo cp /tmp/patched.dtb "$ESP/qcs6490-radxa-dragon-q6a.dtb"
sudo reboot
```

Verify: `sudo dmesg | grep 'fastrpc@89000000'`

### 3. Start the API server

```bash
# Systemd (auto-start on boot)
sudo cp scripts/genie-rs.service /etc/systemd/system/
sudo systemctl enable --now genie-rs

# Or manually:
cd ~/llama-4096-v68-model  # CWD must have libQnnHtpV68Skel.so
export QAIRT=$HOME/qairt/2.47.0.260601
export LD_LIBRARY_PATH=$HOME/llama-4096-v68-model
genie-rs serve --host 0.0.0.0:8080
```

### 4. Use the API

```bash
# Basic text generation
curl http://dragon:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder-0.5b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'

# With tool calling
curl http://dragon:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder-0.5b","messages":[{"role":"user","content":"read file /etc/hostname"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]}'

# Streaming
curl http://dragon:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama32-1b","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

### 5. Opencode

Already configured in `~/.config/opencode/opencode.json` as provider `dragon-npu`:

```json
{
  "providerId": "dragon-npu",
  "type": "openai",
  "apiBase": "http://dragon:8080/v1",
  "models": {
    "llama32-1b": { "tool_call": true, "maxOutput": 512 },
    "qwen2.5-coder-0.5b": { "tool_call": true, "maxOutput": 2048 }
  }
}
```

## Key Insight: The CWD Requirement

`libQnnHtp.so` reads `libQnnHtpV68Skel.so` from the **current working directory** when loading the DSP firmware binary (not via `dlopen` — it reads the raw file and sends it to the DSP via fastrpc). genie-rs **must** be invoked from the model directory:

```bash
cd ~/llama-4096-v68-model
genie-rs serve --host 0.0.0.0:8080
```

Uses two model directories with separate `LD_LIBRARY_PATH` and `WorkingDirectory` values in the systemd service. See `scripts/genie-rs.service`.

## Tool Calling Architecture

Since small NPU models (Llama 3.2 1B, Qwen 0.5B) aren't fine-tuned for function calling, genie-rs uses **server-side tool routing**:

1. User message is keyword-matched against tool names and description terms
2. If matched, the server returns an immediate `tool_calls` response
3. The client executes the tool and sends back the result
4. The server forwards the result to the model for text generation

This preserves the OpenAI-compatible API while working with models that can't natively output function calls.

Supported matching:
- Exact tool name match (e.g. `read_file`)
- Underscore-split name parts (e.g. `read` and `file` independently)
- Description keywords longer than 4 characters

## Links

- [ModelScope: radxa/Llama3.2-1B-1024-qairt-v68](https://modelscope.cn/models/radxa/Llama3.2-1B-1024-qairt-v68)
- [Qualcomm QAIRT docs](https://docs.qualcomm.com/doc/80-63442-10)
- [onnx2qnn — HTP context binary pipeline](https://github.com/ImRIzo/onnx2qnn)

## License

MIT — see [LICENSE](LICENSE).
