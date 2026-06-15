#!/bin/bash
# Deploy dragon-npu-api repo contents to the Dragon board
# Usage: bash deploy-to-dragon.sh <ssh-target> [remote-home-dir]
# Example: bash deploy-to-dragon.sh daniel@dragon /home/daniel

DRAGON="${1:?Usage: $0 <ssh-target> [remote-home-dir]}"
HOME_DIR="${2:-/home/$(whoami)}"

verify() {
  ssh "${DRAGON}" "test -d \"$1\"" || { echo "ERROR: $1 does not exist on ${DRAGON}"; exit 1; }
}

echo "=== Deploy dragon-npu-api ==="
echo "SSH target: ${DRAGON}"
echo "Remote home: ${HOME_DIR}"
echo ""

echo "[1/4] Copy NPU DMA fix overlay..."
scp npu-dma-fix/npu-dma-fix-overlay.dts "${DRAGON}:${HOME_DIR}/npu-dma-fix.dts"
scp npu-dma-fix/build-and-apply.sh "${DRAGON}:${HOME_DIR}/build-npu-fix.sh"
echo "  Copied"

echo "[2/4] Copy reference configs..."
scp genie-configs/htp-model-config-llama32-1b-gqa.json "${DRAGON}:${HOME_DIR}/"
scp genie-configs/htp_backend_ext_config.json "${DRAGON}:${HOME_DIR}/"
echo "  Copied"

echo "[3/4] Copy genie-rs service file..."
scp scripts/genie-rs.service "${DRAGON}:${HOME_DIR}/genie-rs.service"
echo "  Copied (install: sudo cp ${HOME_DIR}/genie-rs.service /etc/systemd/system/)"

echo "[4/4] Copy documentation..."
ssh "${DRAGON}" "mkdir -p ${HOME_DIR}/dragon-npu-api-docs"
scp -r docs/* "${DRAGON}:${HOME_DIR}/dragon-npu-api-docs/"
echo "  Copied"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "On Dragon, to apply NPU DMA fix:"
echo "  sudo bash ~/build-npu-fix.sh"
echo ""
echo "To install systemd service:"
echo "  sudo cp ~/genie-rs.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now genie-rs"
