# Model Build Pipeline — ONNX → HTP Context Binary

How to convert a HuggingFace model to a runnable HTP context binary for the Qualcomm Hexagon v68 NPU (QCS6490).

Tested with **Qwen2.5-Coder-0.5B**. The same process should work for any decoder-only transformer in the 0.5B–1B range.

## Overview

```
HuggingFace → ONNX → QNN model (.cpp + .bin) → aarch64 .so → HTP context binary (.serialized.bin)
                 │                        │                      │
            Step 1                    Step 2                 Step 3
```

## Prerequisites

- **Host (x86)**: QAIRT 2.47 SDK with Python venv
- **Target (aarch64/Dragon)**: QAIRT 2.47 SDK, g++, objcopy, `qnn-context-binary-generator`

## Step 1: ONNX Export

Export from HuggingFace with past key-value states as explicit inputs:

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

n_layers = config.num_hidden_layers        # 24
n_kv_heads = config.num_key_value_heads    # 2
head_dim = config.hidden_size // config.num_attention_heads  # 64

# Generate dummy inputs matching the model's expected shapes
# past_key_values has shape (layers, 2, n_kv_heads, past_len, head_dim)
past_len = 4096
seq_len = 1

input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
attention_mask = torch.ones(1, seq_len + past_len, dtype=torch.long)
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

past_key_values = ()
for _ in range(n_layers):
    k = torch.randn(1, n_kv_heads, past_len, head_dim)
    v = torch.randn(1, n_kv_heads, past_len, head_dim)
    past_key_values = past_key_values + ((k, v),)

torch.onnx.export(
    model, (input_ids, attention_mask, position_ids, past_key_values),
    "model.onnx",
    input_names=["input_ids", "attention_mask", "position_ids"]
        + [f"past_key_values.{i}.{t}" for i in range(n_layers) for t in ("key", "value")],
    output_names=["logits"]
        + [f"present.{i}.{t}" for i in range(n_layers) for t in ("key", "value")],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "mask_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        **{f"past_key_values.{i}.{t}": {0: "batch", 2: "past_len"}
           for i in range(n_layers) for t in ("key", "value")},
    },
    opset_version=17,
)
```

## Step 2: ONNX → QNN Model (with INT8 Quantization)

Use `qnn-onnx-converter` with explicit static dimensions for all 49 inputs:

```bash
qnn-onnx-converter \
    --input_network model.onnx \
    -o /tmp/qwen_qnn \
    -d input_ids 1,1 \
    -d attention_mask 1,4097 \
    -d position_ids 1,1 \
    $(for i in $(seq 0 23); do echo -n \
      "-d past_key_values.$i.key 1,2,4096,64 -d past_key_values.$i.value 1,2,4096,64 "; done) \
    --input_list /tmp/calib/input_list.txt \
    --param_quantizer tf \
    --act_quantizer tf \
    --act_bitwidth 8 \
    --weights_bitwidth 8
```

### Calibration Data (`--input_list.txt`)

The quantizer needs sample input data. All 49 tensor files must match the model's static dimensions (1, 2, 4096, 64 for KV cache). **Use absolute paths** in `input_list.txt`:

```python
import struct, random
calib_dir = "/tmp/calib"
os.makedirs(calib_dir, exist_ok=True)

# input_ids: int32, shape (1,1) => 4 bytes
with open(f"{calib_dir}/input_ids.raw", "wb") as f:
    f.write(struct.pack("I", 1))

# attention_mask: int32, shape (1,4097) => 16388 bytes
with open(f"{calib_dir}/attention_mask.raw", "wb") as f:
    f.write(struct.pack("I" * 4097, *([1] * 4097)))

# position_ids: int32, shape (1,1) => 4 bytes
with open(f"{calib_dir}/position_ids.raw", "wb") as f:
    f.write(struct.pack("I", 0))

# KV cache: float32, shape (1,2,4096,64) each, 524,288 values per tensor
for i in range(24):
    for k in ("key", "value"):
        sz = 2 * 4096 * 64
        data = struct.pack("f" * sz, *([random.uniform(-0.1, 0.1) for _ in range(sz)]))
        with open(f"{calib_dir}/past_key_values.{i}.{k}.raw", "wb") as f:
            f.write(data)

# input_list.txt with ABSOLUTE paths (relative paths cause "Failed to open input file")
names = ["input_ids", "attention_mask", "position_ids"]
for i in range(24):
    names.append(f"past_key_values.{i}.key")
    names.append(f"past_key_values.{i}.value")
with open(f"{calib_dir}/input_list.txt", "w") as f:
    f.write(" ".join(f"{calib_dir}/{n}.raw" for n in names))
```

## Step 3: Build the Model Library (aarch64)

The converter outputs:
- `qwen_qnn.cpp` (~5MB, 2049 tensor/node functions)
- `qwen_qnn.bin` (FP32: 2.4GB; INT8: ~600MB)

### Extract weight files and objcopy

```bash
# The .bin is a tar archive
tar -xf qwen_qnn.bin
mkdir -p binary_obj

# objcopy each .raw to a .o for ARM64
for f in *.raw; do
    objcopy -I binary -O elf64-littleaarch64 -B aarch64 \
        "$f" "binary_obj/${f}.o"
done
rm -f *.raw
```

### Fix support headers

The `-include jni/qnn_config.h` flag (used by QNN SDK's default build) defines `QNN_API` via `#define QNN_API __attribute__((visibility("default")))`. However, the include guard `QNN_CONFIG_H` in `qnn_config.h` **collides** with `QnnGlobalConfig.h` (same guard). Remove the `-include` from the build flags.

Also fix `BINVARSTART`/`BINLEN` prefix in `QnnWrapperUtils.hpp`:
```cpp
// SDK default:
//   extern const uint8_t _binary_obj_binary_##NAME##_raw_start[];
// Fixed (matches actual objcopy output):
//   extern const uint8_t _binary_##NAME##_raw_start[];
```

Replace `_binary_obj_binary_` with `_binary_` in `QnnWrapperUtils.hpp`.

### Compile and link

```bash
g++ -c -std=c++11 -fPIC -O3 -fvisibility=hidden \
    -Ijni/ -I$QNN/include/QNN \
    jni/qwen_model.cpp -o obj/qwen_model.o
g++ -c -std=c++11 -fPIC -O3 -fvisibility=hidden \
    -Ijni/ -I$QNN/include/QNN \
    jni/QnnModel.cpp -o obj/QnnModel.o
g++ -c -std=c++11 -fPIC -O3 -fvisibility=hidden \
    -Ijni/ -I$QNN/include/QNN \
    jni/QnnModelPal.cpp -o obj/QnnModelPal.o
g++ -c -std=c++11 -fPIC -O3 -fvisibility=hidden \
    -Ijni/ -I$QNN/include/QNN \
    jni/QnnWrapperUtils.cpp -o obj/QnnWrapperUtils.o

g++ -shared -fPIC -fvisibility=hidden \
    -o libs/libQwenQnn.so \
    obj/*.o binary_obj/*.o
```

## Step 4: Generate HTP Context Binary

```bash
export LD_LIBRARY_PATH=$QNN/lib/aarch64-oe-linux-gcc11.2

qnn-context-binary-generator \
    --model libs/libQwenQnn.so \
    --backend libQnnHtp.so \
    --output_dir /tmp/qwen_htp \
    --binary_file qwen_htp \
    --config_file config_htp.json \
    --htp_socs qcs6490
```

Config file (`config_htp.json`):
```json
{
    "backend_extensions": {
        "config_file_path": "/path/to/htp_backend_ext_config.json"
    }
}
```

## Known Op Compatibility Issues

### IsNaN (not supported on HTP V68)

The HTP backend does not support `IsNan` operations (used for NaN-mitigation in softmax). These are dead code — the IsNaN output is never consumed.

**Fix**: Remove the IsNaN calls from the `composeGraphs` function in the model .cpp. Also reconnect downstream ops that reference the removed tensor:
- Remove `VALIDATE(addNode__model_layers_X_self_attn_IsNaN(qwen_model), err);`
- Reconnect `MatMul_1` inputs from `Where_3_output` → `Softmax_output`

### BOOL_8 tensors (use UFIXED_POINT_8 instead)

HTP V68's Gather op doesn't accept `BOOL_8` data type.

**Fix**: Replace tensor `dataType` only (not scalar params):
```
.dataType= QNN_DATATYPE_BOOL_8        → .dataType= QNN_DATATYPE_UFIXED_POINT_8
{.scalarParam= (Qnn_Scalar_t) {BOOL… → {.scalarParam= (Qnn_Scalar_t) {BOOL… (keep)
```
| Model | FP32 .bin | INT8 .bin | .so | HTP context binary |
|-------|-----------|-----------|-----|-------------------|
| Qwen2.5-Coder-0.5B | 2.4 GB | 602 MB | 603 MB | 619 MB |
| Qwen2.5-Coder-1.5B | 6.7 GB | 1.7 GB | 1.7 GB | ~2 GB |
| Llama 3.2 1B | — | — | — | 1.7 GB |
- The `IsNaN` fix above eliminates the BOOL_8 → UFIXED_POINT_8 collision
- If Gather still fails on indices, check that the index input tensor uses `INT_32` or `UINT_32`


## Performance

### Model sizes

| Model | FP32 .bin | INT8 .bin | .so | HTP context binary |
|-------|-----------|-----------|-----|-------------------|
| Qwen2.5-Coder-0.5B | 2.4 GB | 602 MB | 603 MB | 619 MB |
| Qwen2.5-Coder-1.5B | 6.7 GB | 1.7 GB | 1.7 GB | ~2 GB |
| Llama 3.2 1B | — | — | — | 1.7 GB |

### Inference speed (on Dragon Q6A NPU v68)

| Model | Tokens/second | Notes |
|-------|--------------|-------|
| Qwen2.5-Coder-0.5B | ~6-8 tok/s | 32768 context |
| Qwen2.5-Coder-1.5B | ~6-8 tok/s | 32768 context, same speed as 0.5B (NPU is compute-bound) |
| Llama 3.2 1B | ~6-8 tok/s | 4096 context |

## Reference: Pre-compiled Models

Pre-compiled HTP context binaries for Qwen2.5-Coder-0.5B already exist at:
- `/home/daniel/Qwen2.5-0.5B-v68/qwen-compiled.serialized.bin` (310 MB, INT8, 32768 ctx)

The pipeline described here produces a **619 MB** binary (same model, same architecture). The size difference comes from quantization granularity (4-bit vs 8-bit or different weight sharing). Both work identically for inference.
