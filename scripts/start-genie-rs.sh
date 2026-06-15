#!/bin/bash
export LD_LIBRARY_PATH=/home/daniel/qairt/2.47.0.260601/lib/aarch64-oe-linux-gcc11.2:/home/daniel/qairt/2.47.0.260601/lib/aarch64-ubuntu-gcc9.4:/home/daniel/qairt/2.47.0.260601/lib/hexagon-v68/unsigned:/home/daniel/llama-4096-v68-model
export RUST_LOG=info
exec /home/daniel/source/dragon-ai/target/release/genie-rs serve --host 0.0.0.0:8080 --registry /home/daniel/source/dragon-ai/models/registry.toml
