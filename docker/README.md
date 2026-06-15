# Dragon NPU AI — Hybrid Pipeline Build System

## Prerequisites

- **Docker** with Buildx support (included in Docker 24+)
- **QAIRT SDK 2.47.0** at `/home/daniel/qairt/2.47.0.260601`
- **dragon-npu-api repo** at `/home/daniel/source/dragon-convert/dragon-npu-api`

## Build libGenie.so with HybridDialog

```bash
# Build the Docker image
docker compose -f docker/compose.yml build genie-builder

# Run the build (produces /opt/qairt/lib/aarch64-linux-gnu/libGenie.so)
docker compose -f docker/compose.yml up genie-builder
```

### Output

```
/opt/qairt/lib/aarch64-linux-gnu/libGenie.so
```

Ship to Dragon:

```bash
scp /opt/qairt/lib/aarch64-linux-gnu/libGenie.so \
    daniel@dragon:/home/daniel/qairt/2.47.0.260601/lib/aarch64-oe-linux-gcc11.2/
ssh daniel@dragon sudo systemctl restart genie-rs
```

## Build MLP HTP Context Binaries

Requires the x86_64 QNN tools from the QAIRT SDK:

```bash
docker compose -f docker/compose.yml run --rm mlp-builder
```

This runs the full pipeline:
1. Export 36 MLP-only ONNX graphs from the Qwen 3B model
2. Convert each to QNN INT8 format
3. Generate HTP context binaries (with `--vtcm_override 0`)

### Output

```
/workspace/mlp_htp_novtcm/qwen_mlp_*.QCS6490.bin  (36 files, 65 MB each)
```

## Interactive Development

```bash
docker compose -f docker/compose.yml run --rm dev
```

## Architecture

The Dockerfile uses a multi-stage build:

```
Stage 1 (genie-builder): Ubuntu 22.04 + GCC 11.2 + GenieDialog build
  → Produces libGenie.so with HybridDialog support
  → Matches QAIRT extensions library ABI (GCC 11.2)

Stage 2 (mlp-builder): genie-builder + Python ML deps
  → Exports ONNX graphs from HuggingFace models
  → Converts to QNN INT8 quantized format
  → Generates HTP context binaries

Stage 3 (runtime): Lightweight runtime deps for Dragon
```

## Why Ubuntu 22.04?

The QAIRT SDK 2.47.0 extensions library (`libQairtHtpBackendExtensions.so`) was compiled with **GCC 11.2** for the `aarch64-oe-linux-gcc11.2` target. Building the HybridDialog code with the same GCC version ensures C++ ABI compatibility for:
- Virtual function table layout (IBackend interface)
- `std::string` small string optimization
- Exception handling frames
- Name mangling

Ubuntu 22.04 ships with **GCC 11.4** as the default compiler, which is forward-compatible with GCC 11.2.
