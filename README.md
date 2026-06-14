# Dragon NPU API — Llama 3.2 1B on Hexagon v68

Run an OpenAI-compatible chat completion API on the **Radxa Dragon Q6A** board's **Qualcomm Hexagon v68 NPU** (QCS6490).

- **Model**: Llama 3.2 1B — pre-compiled QNN HTP context binary from ModelScope
- **Runtime**: QAIRT 2.47 / Genie SDK — Qualcomm's on-device LLM inference
- **API**: OpenAI-compatible chat completions (`/v1/chat/completions`)
- **Performance**: ~20 tok/s on NPU, 1024 context window

## Repo Contents

| Path | Description |
|------|-------------|
| `genie-rs/` | **Complete genie-rs Rust source** — build your own API server |
| `genie-rs/src/main.rs` | CLI, registry loading, `--working-dir` support |
| `genie-rs/src/config/registry.rs` | Model registry deserialization with HTP fields |
| `genie-rs/src/config/qnn_config.rs` | JSON config generator for Genie SDK |
| `genie-rs/src/context/genie_context.rs` | GenieDialog C API wrapper (create, query, tokenize) |
| `genie-rs/src/server/` | axum HTTP router + OpenAI-compatible handlers |
| `genie-rs/src/ffi/mod.rs` | FFI imports from Genie C headers |
| `genie-rs/models/registry.toml` | Model registry — Qwen GGUF + Llama HTP entries |
| `genie-rs/build.rs` | bindgen + rpath for QAIRT 2.47 |
| `genie-rs/Cargo.toml` | Rust dependencies |
| `docs/01-overview.md` | Full architecture walkthrough |
| `docs/02-npu-dma-fix.md` | NPU DMA fix — DTB patching for fastrpc |
| `docs/03-genie-rs-api.md` | API server setup and configuration |
| `docs/04-htp-config.md` | HTP backend configuration reference |
| `docs/05-kernel-update-procedure.md` | What to do after a kernel update |
| `docs/06-troubleshooting.md` | Common issues and solutions |
| `npu-dma-fix/` | DTBO overlay source + build-and-apply script |
| `genie-configs/` | Reference HTP config files (from ModelScope model) |
| `scripts/genie-rs.service` | Systemd unit for auto-start |
| `scripts/start-genie-server.sh` | Manual startup script |
| `scripts/deploy-to-dragon.sh` | Deploy repo to Dragon via SCP |
| `LINKS.md` | External resources |

## Quick Start

### Prerequisites

- Radxa Dragon Q6A with Ubuntu 24.04, kernel 6.18.2-4-qcom
- QAIRT 2.47 SDK at `~/qairt/2.47.0.260601/`
- Rust toolchain (on Dragon)
- ModelScope model: `radxa/Llama3.2-1B-1024-qairt-v68` at `~/llama-v68-model/`
- `git-lfs` for model download

### 1. Build genie-rs (on Dragon)

```bash
cd genie-rs
source ~/.cargo/env
export QAIRT=$HOME/qairt/2.47.0.260601
cargo build --release
```

### 2. Apply NPU DMA fix

```bash
sudo apt install device-tree-compiler

# Build DTBO
dtc -@ -I dts -O dtb -o npu-dma-fix.dtbo npu-dma-fix-overlay.dts

# Merge into stock DTB (adjust kernel version if different)
ESP="/boot/efi/RadxaOS/6.18.2-4-qcom"
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb" /tmp/orig.dtb
fdtoverlay -i /tmp/orig.dtb -o /tmp/patched.dtb npu-dma-fix.dtbo

# Backup stock and deploy patched
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb"{,.stock}
sudo cp /tmp/patched.dtb "$ESP/qcs6490-radxa-dragon-q6a.dtb"

sudo reboot
```

Verify: `sudo dmesg | grep 'fastrpc@89000000'`

### 3. Start the API server

```bash
export QAIRT=$HOME/qairt/2.47.0.260601
export LD_LIBRARY_PATH=$HOME/llama-v68-model
cd $HOME/llama-v68-model    # <-- REQUIRED: CWD must have libQnnHtpV68Skel.so
genie-rs serve --host 0.0.0.0:8080 --registry /path/to/genie-rs/models/registry.toml
```

Or use the systemd service:
```bash
sudo cp scripts/genie-rs.service /etc/systemd/system/
sudo systemctl enable --now genie-rs
```

### 4. Use the API

```bash
curl http://dragon:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama32-1b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

## Key Insight: The CWD Requirement

`libQnnHtp.so` reads `libQnnHtpV68Skel.so` from the **current working directory** when loading the DSP firmware binary (not via `dlopen` — it reads the raw file and sends it to the DSP via fastrpc). Both `genie-t2t-run` and genie-rs **must** be invoked from the model directory:

```bash
cd ~/llama-v68-model    # <-- required
genie-rs serve ...
```

The systemd service handles this via `WorkingDirectory=/home/daniel/llama-v68-model`.

## Links

- [ModelScope: radxa/Llama3.2-1B-1024-qairt-v68](https://modelscope.cn/models/radxa/Llama3.2-1B-1024-qairt-v68)
- [Qualcomm QAIRT docs](https://docs.qualcomm.com/doc/80-63442-10)
- Foadsf's NPU DMA fix gist — **FIXME: search GitHub gists for "Foadsf fastrpc DMA"**

## License

MIT — see [LICENSE](LICENSE).
