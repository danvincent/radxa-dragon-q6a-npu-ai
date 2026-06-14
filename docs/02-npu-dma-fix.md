# NPU DMA Fix — DTB Patching for fastrpc

## Problem

QNN HTP backend fails with `Error 14001 (AEE_EUNSUPPORTED)`. Root cause: the fastrpc kernel driver has no reserved DMA pool for communicating with the NPU (CDSP/ADSP firmware).

```
[    3.656906] qcom,fastrpc: no reserved DMA memory for FASTRPC
```

## Root Cause

The Device Tree loaded by systemd-boot is missing:
1. A `shared-dma-pool` reserved-memory node for fastrpc
2. `memory-region` property on the CDSP and ADSP fastrpc nodes

## Discovery

This issue was first identified by **Foadsf** in a GitHub gist documenting the same problem on similar Qualcomm platforms (RB5, SA8295P, Dragon). The gist identified the `no reserved DMA memory for FASTRPC` kernel message and proposed DT overlay solutions.

- Gist: [Foadsf's Qualcomm fastrpc DMA fix gist](https://gist.github.com/Foadsf/...) — **FIXME: Replace `...` with actual gist URL. Search for "Foadsf fastrpc DMA".**

## Solution Strategy

The Dragon Q6A uses **UEFI + systemd-boot** (not U-Boot). The DTB is loaded from the EFI System Partition. Our solution:

1. Create a Device Tree Overlay (DTBO) that adds the missing DMA pool and properties
2. Merge it into the stock DTB using `fdtoverlay`
3. Deploy the patched DTB to the ESP

### Why not other approaches?

| Approach | Why it failed |
|----------|---------------|
| **extlinux `fdt`** | extlinux only configures U-Boot; Dragon uses UEFI + systemd-boot |
| **`/boot/dtbo/` overlay** | Radxa's overlay loader only supports symbol-based (`__fixups__`) overlays, not `target-path` |
| **Round-trip DTS → DTB** | Decompiling/compiling introduces warnings and potential errors |

## The DTBO Overlay

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/reserved-memory";
        __overlay__ {
            #address-cells = <0x02>;
            #size-cells = <0x02>;
            ranges;

            fastrpc_dma_pool: fastrpc@89000000 {
                compatible = "shared-dma-pool";
                reg = <0x00 0x89000000 0x00 0x02000000>;
                no-map;
            };
        };
    };

    fragment@1 {
        target-path = "/soc@0/remoteproc@a300000/glink-edge/fastrpc";
        __overlay__ {
            memory-region = <&fastrpc_dma_pool>;
        };
    };

    fragment@2 {
        target-path = "/soc@0/remoteproc@3700000/glink-edge/fastrpc";
        __overlay__ {
            memory-region = <&fastrpc_dma_pool>;
        };
    };
};
```

### Pool Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Address | `0x89000000` | Between `cdsp-secure-heap@82700000` (ends 0x82710000) and `adsp@8b800000` (starts 0x8b800000) — ~144 MB gap |
| Size | `0x02000000` (32 MB) | Sufficient for fastrpc DMA operations |
| Type | `shared-dma-pool` | Standard DMA pool for shared memory between CPU and DSP |
| Mapping | `no-map` | Don't create virtual memory mapping (reserved for device) |

## Application Procedure

### One-time setup

```bash
# Build DTBO
dtc -@ -I dts -O dtb -o npu-dma-fix.dtbo npu-dma-fix-overlay.dts

# Merge with stock DTB
ESP="/boot/efi/RadxaOS/6.18.2-4-qcom"
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb" /tmp/orig.dtb
fdtoverlay -i /tmp/orig.dtb -o /tmp/patched.dtb npu-dma-fix.dtbo

# Backup and deploy
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb"{,.stock}
sudo cp /tmp/patched.dtb "$ESP/qcs6490-radxa-dragon-q6a.dtb"

# Reboot
sudo reboot
```

### Verification

```bash
# Check kernel logs
sudo dmesg | grep -E 'fastrpc|dma pool|reserved memory'

# Expected output:
# [    0.000000] Reserved memory: created DMA memory pool at 0x0000000089000000, size 32 MiB
# [    0.000000] OF: reserved mem: initialized node fastrpc@89000000, compatible id shared-dma-pool
# [    0.000000] OF: reserved mem: 0x0000000089000000..0x000000008affffff (32768 KiB) nomap non-reusable fastrpc@89000000
# [    3.715926] qcom,fastrpc ... assigned reserved memory node fastrpc@89000000

# Check device tree
find /proc/device-tree -path '*reserved-memory*' -name 'fastrpc*'
ls /proc/device-tree/soc@0/remoteproc@*/glink-edge/fastrpc/memory-region
```

The `no reserved DMA memory for FASTRPC` message must be **absent**.

## Design Notes

- **Phandle**: `fdtoverlay` auto-assigns the phandle (was `0x2b5` in our case — the next available after the stock DTB's max of `0x2b4`)
- **Skel loading**: After the DMA fix, `libQnnHtp.so` can load `libQnnHtpV68Skel.so` onto the DSP via fastrpc. If fastrpc sessions still fail, check DSP firmware compatibility.

## Recovery

```bash
# Restore stock DTB
ESP="/boot/efi/RadxaOS/6.18.2-4-qcom"
sudo cp "$ESP/qcs6490-radxa-dragon-q6a.dtb"{.stock,}
sudo reboot
```

All scripts are in the `npu-dma-fix/` directory of this repo.
