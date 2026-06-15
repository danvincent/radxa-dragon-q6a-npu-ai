#!/bin/bash
# Start genie-rs server manually (for testing)
# Must be run from the model directory
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-}"
REGISTRY="${REGISTRY:-./models/registry.toml}"
GENIE_BIN="${GENIE_BIN:-./target/release/genie-rs}"

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: MODEL_DIR=/path/to/model $0"
    echo "Or set MODEL_DIR environment variable"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$GENIE_BIN" ]; then
    echo "ERROR: genie-rs binary not found: $GENIE_BIN"
    echo "Build it first:"
    echo "  cargo build --release"
    exit 1
fi

cd "$MODEL_DIR"

export QAIRT="${QAIRT:-/opt/qairt}"
export LD_LIBRARY_PATH="$MODEL_DIR"

echo "=== Starting genie-rs ==="
echo "  Working directory: $(pwd)"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  QAIRT: $QAIRT"
echo "  Registry: $REGISTRY"
echo ""

exec "$GENIE_BIN" serve --host 0.0.0.0:8080 --registry "$REGISTRY"
