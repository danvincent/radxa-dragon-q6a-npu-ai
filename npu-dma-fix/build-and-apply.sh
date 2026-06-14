#!/bin/bash
# Build DTBO and apply to ESP
# Usage: sudo bash build-and-apply.sh <kernel-version>
# Example: sudo bash build-and-apply.sh 6.18.2-4-qcom
set -euo pipefail

KERNEL_VER="${1:-6.18.2-4-qcom}"
ESP="/boot/efi/RadxaOS/${KERNEL_VER}"

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

echo "=== NPU DMA Fix — Build and Apply for kernel ${KERNEL_VER} ==="

# 1. Build DTBO
echo ""
echo "[1/4] Building DTBO overlay..."
dtc -@ -I dts -O dtb -o /tmp/npu-dma-fix.dtbo npu-dma-fix-overlay.dts
echo "  ✅ DTBO built ($(stat -c%s /tmp/npu-dma-fix.dtbo) bytes)"

# 2. Backup stock DTB
echo ""
echo "[2/4] Backing up stock DTB..."
if [ ! -f "${ESP}/qcs6490-radxa-dragon-q6a.dtb.stock" ]; then
    cp "${ESP}/qcs6490-radxa-dragon-q6a.dtb" "${ESP}/qcs6490-radxa-dragon-q6a.dtb.stock"
    echo "  ✅ Stock DTB backed up"
else
    echo "  ℹ️  Stock backup already exists, skipping"
fi

# 3. Merge overlay
echo ""
echo "[3/4] Merging overlay into DTB..."
cp "${ESP}/qcs6490-radxa-dragon-q6a.dtb" /tmp/orig.dtb
fdtoverlay -i /tmp/orig.dtb -o /tmp/patched.dtb /tmp/npu-dma-fix.dtbo
cp /tmp/patched.dtb "${ESP}/qcs6490-radxa-dragon-q6a.dtb"
echo "  ✅ Patched DTB deployed ($(stat -c%s "${ESP}/qcs6490-radxa-dragon-q6a.dtb") bytes)"

# 4. Verify
echo ""
echo "[4/4] Verifying..."
dtc -I dtb -O dts /tmp/patched.dtb 2>/dev/null | grep -q 'fastrpc@89000000' && echo "  ✅ DMA pool node present" || {
    echo "  ❌ DMA pool node NOT FOUND!"
    exit 1
}
echo ""
echo "=== Done. Reboot to apply:  sudo reboot ==="
