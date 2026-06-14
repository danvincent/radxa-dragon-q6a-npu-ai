# Troubleshooting

## Error 14001 (AEE_EUNSUPPORTED)

### DMA Pool Not Present

Check kernel logs:
```bash
sudo dmesg | grep -E 'fastrpc|dma pool|reserved memory'
```

If you see `no reserved DMA memory for FASTRPC`, the DTB fix hasn't been applied or was overwritten by a kernel update. See `docs/02-npu-dma-fix.md` or `docs/05-kernel-update-procedure.md`.

### DMA Pool Present but Still Failing

If the pool is allocated but `GenieDialog_create` returns 14001:

```
fastrpc_apps_user.c:1724: remote_handle64_open: Successfully opened handle
[INFO] "Using create From Binary List Async"
```

But then fails — the DSP firmware might not support the session type. This is rare with the ModelScope model (it's pre-compiled for v68), but can happen with custom context binaries.

**Check DSP firmware**:
```bash
# List compute contexts
ls /proc/device-tree/soc@0/remoteproc@a300000/glink-edge/fastrpc/
# Should show compute-cb@1 through compute-cb@14 for CDSP
```

**Try the model's own genie-t2t-run**:
```bash
cd ~/llama-v68-model
LD_LIBRARY_PATH=. ./genie-t2t-run -c htp-model-config-llama32-1b-gqa.json -p "Hello"
```

If this works but genie-rs doesn't, compare the JSON configs. The generated config must match the reference config exactly in the critical HTP fields (`kv-dim`, `poll`, `allow-async-init`, positional encoding).

## genie-rs Fails to Start

### Address Already in Use

```bash
ss -tlnp | grep 8080
sudo systemctl restart genie-rs
```

### Library Not Found

```bash
# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
# Must include: /home/daniel/llama-v68-model

# Check QAIRT
echo $QAIRT
# Must be: /home/daniel/qairt/2.47.0.260601

# Check actual libraries
ls -la ~/llama-v68-model/libQnnHtp.so
```

### Wrong Working Directory

```bash
# genie-rs must run from the model directory
pwd  # Must be: /home/daniel/llama-v68-model
ls libQnnHtpV68Skel.so  # Must be accessible from CWD
```

## genie-t2t-run Works, genie-rs Doesn't

1. **CWD**: Both must be run from the model directory (see `docs/03-genie-rs-api.md` for why)
2. **Registry paths**: Use absolute paths in `registry.toml` for `tokenizer` and `ctx_bins`
3. **Config difference**: Run genie-t2t-run with `--print-config` (if available) and compare to what genie-rs generates
4. **LD_LIBRARY_PATH**: Ensure both have the same environment

## Model Output Garbled

### Wrong Tokenizer

Ensure the tokenizer config matches the model:
- Llama 3.2 1B uses `tokenizer.json` (tiktoken-based, 128256 vocab)
- Qwen 2.5 uses `qwen.tiktoken` (151936 vocab)

Token IDs for Llama 3:
- BOS: 128000, EOS: 128009, PAD: 128004

### Wrong Chat Template

The chat template must match the model's training format:
- Llama 3: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
- Qwen 2.5: `<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n`

## Kernel Update Kerfuffle

See `docs/05-kernel-update-procedure.md`. TL;DR: after every `apt upgrade` that includes a new kernel:

```bash
KERNEL_VER=$(uname -r)  # new version after reboot
ESP="/boot/efi/RadxaOS/${KERNEL_VER}"
fdtoverlay -i "${ESP}/qcs6490-radxa-dragon-q6a.dtb" -o /tmp/patched.dtb npu-dma-fix.dtbo
sudo cp /tmp/patched.dtb "${ESP}/qcs6490-radxa-dragon-q6a.dtb"
sudo reboot
```

## Cannot Access ESP Partition

```bash
# Check mount
mount | grep /boot/efi
# Should show /dev/nvme0n1p2 on /boot/efi

# If not mounted
sudo mount /dev/nvme0n1p2 /boot/efi
```

## systemd Service Won't Start

```bash
# Check status
sudo systemctl status genie-rs

# View full log
sudo journalctl -u genie-rs -n 50 --no-pager

# Try running manually (as daniel)
cd ~/llama-v68-model
export QAIRT=$HOME/qairt/2.47.0.260601
export LD_LIBRARY_PATH=$HOME/llama-v68-model
~/source/dragon-ai/target/release/genie-rs serve --host 0.0.0.0:8080 --registry ~/source/dragon-ai/models/registry.toml
```

## Check if NPU is Actually Being Used

```bash
# Monitor DSP activity
cat /sys/kernel/debug/fastrpc/domains

# Check power management of CDSP
cat /sys/devices/platform/soc@0/a300000.remoteproc/remoteproc/remoteproc0/state
# Should be "running" during inference
```
