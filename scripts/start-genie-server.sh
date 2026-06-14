#!/bin/bash
# Start genie-rs server manually (for testing)
# Must be run from the model directory
set -euo pipefail

MODEL_DIR="/home/daniel/llama-v68-model"
REGISTRY="/home/daniel/source/dragon-ai/models/registry.toml"
GENIE_BIN="/home/daniel/source/dragon-ai/target/release/genie-rs"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$GENIE_BIN" ]; then
    echo "ERROR: genie-rs binary not found: $GENIE_BIN"
    echo "Build it first:"
    echo "  cd ~/source/dragon-ai && source ~/.cargo/env && cargo build --release"
    exit 1
fi

cd "$MODEL_DIR"

export QAIRT="${QAIRT:-$HOME/qairt/2.47.0.260601}"
export LD_LIBRARY_PATH="$MODEL_DIR"

echo "=== Starting genie-rs ==="
echo "  Working directory: $(pwd)"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  QAIRT: $QAIRT"
echo "  Registry: $REGISTRY"
echo ""

exec "$GENIE_BIN" serve --host 0.0.0.0:8080 --registry "$REGISTRY"
