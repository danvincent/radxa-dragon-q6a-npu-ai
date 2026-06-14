# Kernel Update Procedure

When the Dragon receives a kernel update (e.g., after `apt upgrade`), the kernel image and DTB on the EFI System Partition are replaced. The NPU DMA fix **must be re-applied** because the DTB is overwritten.

## What Changes During a Kernel Update

1. New kernel image at `/boot/efi/RadxaOS/<new-version>/linux`
2. New initrd at `/boot/efi/RadxaOS/<new-version>/initrd.img-<new-version>`
3. **New DTB** at `/boot/efi/RadxaOS/<new-version>/qcs6490-radxa-dragon-q6a.dtb`
4. New systemd-boot entry at `/boot/efi/loader/entries/RadxaOS-<new-version>.conf`
5. The old kernel directory may be kept

The stock DTB from the new package will NOT have the fastrpc DMA pool. The NPU will stop working until the fix is re-applied.

## Step-by-Step: After Kernel Update

### 1. Identify the new kernel version

```bash
ls /boot/efi/RadxaOS/
# Example:  6.18.2-4-qcom  6.18.2-5-qcom
```

### 2. Build and merge the DTBO overlay

```bash
# Set the new kernel version
KERNEL_VER="6.18.2-5-qcom"
ESP="/boot/efi/RadxaOS/${KERNEL_VER}"

# Build DTBO (if not already built)
dtc -@ -I dts -O dtb -o npu-dma-fix.dtbo npu-dma-fix-overlay.dts

# Backup the new stock DTB
sudo cp "${ESP}/qcs6490-radxa-dragon-q6a.dtb" /tmp/orig.dtb

# Merge overlay
fdtoverlay -i /tmp/orig.dtb -o /tmp/patched.dtb npu-dma-fix.dtbo

# Save backup and deploy
sudo cp "${ESP}/qcs6490-radxa-dragon-q6a.dtb" "${ESP}/qcs6490-radxa-dragon-q6a.dtb.stock"
sudo cp /tmp/patched.dtb "${ESP}/qcs6490-radxa-dragon-q6a.dtb"
```

### 3. (Optional) Update systemd-boot entry

If the kernel update created a new boot entry, the new entry will be used automatically (systemd-boot uses the highest version). Verify:

```bash
cat /boot/efi/loader/entries/RadxaOS-${KERNEL_VER}.conf
# Should show devicetree pointing to the new kernel dir
```

No changes needed — the DTB filename is the same, we replaced it in-place.

### 4. Reboot and verify

```bash
sudo reboot
# After reboot:
sudo dmesg | grep 'fastrpc@89000000'
```

### 5. If the fix didn't take

```bash
# Check which DTB was loaded
find /proc/device-tree -name 'fastrpc*' -path '*reserved*'

# If the stock DTB was used (no fastrpc DMA pool):
# 1. Check if a separate boot entry is being used
cat /proc/cmdline
# Look for: root=UUID=...

# 2. Re-apply the fix, ensuring the correct ESP directory
```

## Automating the Re-Application

Create a script that runs after every kernel update using a kernel post-install hook:

```bash
#!/bin/bash
# /etc/kernel/postinst.d/npu-dma-fix
# Run after every kernel install to patch the DTB
set -euo pipefail

KERNEL_VER="$1"
ESP="/boot/efi/RadxaOS/${KERNEL_VER}"

if [ ! -d "$ESP" ]; then
    exit 0
fi

DTBO="/home/daniel/npu-dma-fix/npu-dma-fix.dtbo"
ORIG="${ESP}/qcs6490-radxa-dragon-q6a.dtb"
PATCHED="/tmp/npu-patched-${KERNEL_VER}.dtb"

if [ ! -f "$DTBO" ]; then
    echo "NPU DMA fix: DTBO not found at $DTBO, skipping"
    exit 0
fi

# Backup and patch
cp "$ORIG" "${ORIG}.stock"
fdtoverlay -i "$ORIG" -o "$PATCHED" "$DTBO"
cp "$PATCHED" "$ORIG"
rm "$PATCHED"

echo "NPU DMA fix: patched DTB for kernel ${KERNEL_VER}"
```

Make it executable:
```bash
sudo chmod +x /etc/kernel/postinst.d/npu-dma-fix
```

**Caveat**: Custom hooks may be removed by the distribution's kernel package updates. Check after major Ubuntu version upgrades.

## Checking Kernel Versions During Boot

At the systemd-boot menu, the current kernel version is shown in the entry title. Select the correct entry. The NPU fix is applied per-kernel-version; if you boot a kernel without the patched DTB, the NPU will not work.

## Full Recovery

If the NPU stops working and you need to confirm the fix is active:

```bash
# Check DMA pool
sudo dmesg | grep -i 'dma pool'
# Should show: "created DMA memory pool at 0x0000000089000000"

# Check fastrpc
sudo dmesg | grep -i fastrpc
# Should NOT show: "no reserved DMA memory for FASTRPC"

# Check device tree
ls /proc/device-tree/soc@0/remoteproc@*/glink-edge/fastrpc/memory-region
# Should show (or symlink to) the DMA pool node
```

If any of these fail, re-apply the fix for the currently booted kernel.
