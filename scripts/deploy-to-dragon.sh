#!/bin/bash
# Deploy dragon-npu-api repo contents to the Dragon board
# Usage: bash deploy-to-dragon.sh <dragon-host>
# Example: bash deploy-to-dragon.sh daniel@dragon

DRAGON="${1:-daniel@dragon}"

echo "=== Deploy dragon-npu-api to Dragon ==="
echo "Target: ${DRAGON}"

# 1. Copy NPU DMA fix overlay
echo ""
echo "[1/5] NPU DMA fix overlay..."
scp npu-dma-fix/npu-dma-fix-overlay.dts "${DRAGON}:/home/daniel/npu-dma-fix.dts"
scp npu-dma-fix/build-and-apply.sh "${DRAGON}:/home/daniel/build-npu-fix.sh"
echo "  ✅ Copied"

# 2. Copy reference configs
echo ""
echo "[2/5] Reference configs..."
scp genie-configs/htp-model-config-llama32-1b-gqa.json "${DRAGON}:/home/daniel/"
scp genie-configs/htp_backend_ext_config.json "${DRAGON}:/home/daniel/"
echo "  ✅ Copied"

# 3. Copy genie-rs patches
echo ""
echo "[3/5] genie-rs patches..."
scp genie-rs-patches/*.rs "${DRAGON}:/home/daniel/"
echo "  ✅ Copied"

# 4. Copy service file
echo ""
echo "[4/5] Systemd service..."
scp scripts/genie-rs.service "${DRAGON}:/home/daniel/genie-rs.service"
echo "  ✅ Copied (install with: sudo cp ~/genie-rs.service /etc/systemd/system/)"

# 5. Copy docs
echo ""
echo "[5/5] Documentation..."
ssh "${DRAGON}" "mkdir -p /home/daniel/dragon-npu-api-docs"
scp -r docs/* "${DRAGON}:/home/daniel/dragon-npu-api-docs/"
echo "  ✅ Copied"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "On Dragon, to apply NPU DMA fix:"
echo "  sudo bash ~/build-npu-fix.sh"
echo ""
echo "To install systemd service:"
echo "  sudo cp ~/genie-rs.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now genie-rs"
