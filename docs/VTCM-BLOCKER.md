# VTCM Blocker: Hybrid Pipeline Integration on QCS6490 V68

**Status**: Blocked — all infrastructure built, one proprietary SDK dependency remains.

---

## 1. What VTCM Is

**Vector Tightly Coupled Memory (VTCM)** is a small, fast SRAM bank physically inside the Hexagon DSP core (V68 on QCS6490). It is ~4 MB and sits at the L1 level — ~10x lower latency than system DDR accessed over the DMA bus.

The HTP (Hexagon Tensor Processor) NPU inside the DSP uses VTCM as its **working set memory** for matrix operations. When the NPU runs a layer, it must load weights + activations into VTCM before computation starts. If the working set exceeds VTCM, the compiler must spill to DDR — killing throughput.

## 2. How VTCM Gets Reserved

VTCM is a **global, exclusive DSP resource**. Only one piece of code can reserve it at a time. The reservation happens during QNN context creation:

```
QnnContext_createFromBinary(backend, device, binary, size, &ctx, NULL)
                         ↓
                    HTP firmware reads binary header
                         ↓
              "This context needs 4 MB VTCM" (embedded in .bin)
                         ↓
                    Backend checks: is VTCM available?
                         ↓
                    If not configured → returns err 5005
```

The context binary embeds the VTCM request at compile time. The `--vtcm_override N` flag during `qnn-context-binary-generator` sets this value:

- `--vtcm_override 0` → "use SoC maximum" (still 4 MB on QCS6490)
- `--vtcm_override 1` → request 1 MB

There is **no flag to request 0 VTCM** because any HTP graph needs some VTCM for activation buffers even without KV cache.

## 3. The Configuration Gap

The HTP backend (`libQnnHtp.so`) does not know the SoC's physical VTCM size by default. It needs to be told via backend configuration:

```c
QnnBackend_Config_t configs[] = {
    { .option = QNN_BACKEND_CONFIG_OPTION_PLATFORM,
      .platformOption = "SOC_MODEL=35;DSP_ARCH=v68;VTCM_SIZE=4" },
    { .option = QNN_BACKEND_CONFIG_OPTION_TERMINATION }
};
backendCreate(log, configs, &handle);
```

The problem: **the exact key-value string format for `platformOption` is undocumented and proprietary**. It is defined only in the HTP backend firmware code, which Qualcomm does not publish.

## 4. The Extensions Library

Instead of documenting the platform string format, Qualcomm provides **`libQairtHtpBackendExtensions.so`** / **`libQnnHtpNetRunExtensions.so`** — a plugin library that:

1. Reads a **JSON config file** (`htp_backend_ext_config.json`):
   ```json
   {
     "devices": [{"soc_id": 35, "dsp_arch": "v68", ...}],
     "memory": {"mem_type": "shared_buffer"}
   }
   ```

2. Parses it and calls `prepareSoc(0, "v68", 4, "qcs6490", ...)` internally
3. Generates the correct platform option configs via `beforeBackendInitialize()`
4. Returns them so `backendCreate` can use them

The correct flow:

```
backendCreate(log, configs_from_extensions, &handle)
  → HTP backend knows: "I'm on QCS6490 with 4 MB VTCM"
  → contextCreateFromBinary(handle, ..., binary, ...)
  → "Binary wants 4 MB VTCM, I have 4 MB → granted"
  → Context created successfully
```

## 5. Why Direct Calls to the Extensions Library Fail

The extensions library exposes its functionality through a C++ **pure virtual interface** (`IBackend`):

```cpp
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual bool setupLogging(QnnLog_Callback_t, QnnLog_Level_t) = 0;
    virtual bool initialize(void* backendLibHandle) = 0;
    virtual bool loadConfig(std::string configFile) = 0;
    virtual bool beforeBackendInitialize(QnnBackend_Config_t***, uint32_t*) = 0;
    // ... ~50 more virtual methods
};
```

This interface is designed to be called from C++ code compiled with the **same compiler version and C++ runtime** as the extensions library. The extensions library in QAIRT 2.47.0 was compiled with **GCC 11.2 + libstdc++** (for aarch64-oe-linux) or **Clang 14 + libc++** (for x86_64-linux).

On Dragon, we have **GCC 13.3 + libstdc++** (the system compiler). The C++ ABI between GCC 11 and GCC 13 is **source-compatible but not binary-compatible** for:

- Virtual function table layout (potential reordering)
- `std::string` small string optimization (changed between GCC 11 and 13)
- Exception handling frame formats
- Name mangling edge cases

The `IBackend::initialize()` call works (returns true/false), but `loadConfig()` returns true while silently failing to populate internal state. The `beforeBackendInitialize()` returns `count=0` configs instead of the expected SOC/VTCM configs.

**Confirmed by testing:**

1. `loadConfig()` with the same JSON file GenieDialog uses → returns `true`
2. `beforeBackendInitialize()` → returns `count=0`
3. Creating backend with 0 configs → backend exists but does not know SOC/VTCM
4. Loading MLP context → fails with VTCM error 5005

When GenieDialog calls the same functions through the same library, it gets `count>0` configs and MLP loading works — because GenieDialog's `libGenie.so` was compiled by Qualcomm with **GCC 11.2**, matching the extensions library's ABI.

## 6. The GenieDialog Rebuild Problem

The only fix is to rebuild `libGenie.so` with our modifications using a compiler compatible with the extensions library (GCC 11.2 or Clang 14 + libc++).

### Available toolchains

| Platform | Compiler | Std Library | Compatible? |
|----------|----------|-------------|-------------|
| Dragon (OE Linux) | GCC 13.3 | libstdc++ 13 | ❌ ABI mismatch |
| Dragon | Clang 14 + libc++-14 | libc++ | ✅ ABI match |
| x86 (Fedora 44) | GCC 14 | libstdc++ 14 | ❌ ABI mismatch |
| x86 (Fedora 44) | No clang++-14 | — | ❌ |

### Build failures

Using Clang 14 + libc++ on Dragon, the build script compiles ~80 source files. The link step fails because some SDK source files depend on platform-specific headers not available on aarch64 Linux:

| File | Dependency | Issue |
|------|-----------|-------|
| `DxAllocator.cpp` | DirectX buffer allocator | Windows-only, not available on Linux |
| `DxRegs.cpp` | DirectX buffer registration | Windows-only |
| `DmaAllocator.cpp` | DMA buffer allocator | Needs platform DMA headers |
| `DmaRegs.cpp` | DMA buffer registration | Needs platform DMA headers |
| `Accuracy.cpp` | Accuracy checker | Missing internal SDK headers |

These files are part of the example SDK but were designed for Windows/Android builds. They cannot compile on aarch64 Linux without the full QAIRT Linux SDK toolchain (which includes cross-compilation headers for these components).

## 7. How GenieDialog Sidesteps This

GenieDialog's `libGenie.so` was **pre-compiled by Qualcomm** with GCC 11.2. It calls the extensions library with matching ABI. The backend handle (with VTCM configured) lives inside GenieDialog's internal C++ object graph:

```
GenieDialog_create()
  → QnnApi::getQnnInterface(path)          // loads libQnnHtp.so
  → QnnApi::initializeBackendExtensions()  // loads extensions .so
  → extensions->setupLogging()
  → extensions->initialize(backendLibHandle)
  → extensions->loadConfig(configFilePath)       // reads JSON
  → extensions->beforeBackendInitialize(&configs, &count)  // returns VTCM configs!
  → backendCreate(log, configs, &backendHandle)  // creates backend WITH VTCM
  → extensions->afterBackendInitialize()
```

The `backendHandle` is stored as `QnnApi::m_backendHandle` (private member). GenieDialog does **not** expose it through any C API. There is no `GenieDialog_getQnnBackendHandle()` function.

To share this handle, the hybrid pipeline uses `qnn_handles.h/.cpp` — a pair of global variables:

```cpp
extern "C" {
    void* g_qnn_backend = nullptr;  // set by QnnApi::initializeBackend()
    void* g_qnn_device = nullptr;   // set by QnnApi::createDevice()
}
```

The HybridDialog reads these globals during MLP context creation. But this requires recompiling `libGenie.so` with the modified `QnnApi.cpp`, which brings us back to the build problem.

## 8. What Is Needed to Fix This

### Option A: Ubuntu 22.04 aarch64 with g++-11 (Recommended)

Dragon Radxa Q6A's Yocto-based Linux does not have `g++-11` in its repositories. Installing Ubuntu 22.04 (or Debian Bookworm) for aarch64 on the Dragon would provide `g++-11` via `apt`. Then:

1. `apt install g++-11`
2. Compile the modified `libGenie.so` source with GCC 11 (matching the extensions library ABI)
3. Deploy the rebuilt library
4. The HybridDialog uses the shared backend handle → MLP contexts load with VTCM → hybrid pipeline works

### Option B: Official Qualcomm aarch64 Cross-Compiler

The QAIRT SDK for Linux (full version, not just examples) includes an `aarch64-oe-linux-gcc11.2` cross-compiler for x86 hosts. Installing this lets you rebuild `libGenie.so` for aarch64 from an x86 machine with the correct ABI:

```
~/qairt/.../bin/aarch64-oe-linux-gcc11.2/aarch64-oe-linux-g++
```

This cross-compiler is part of the QAIRT Linux SDK package available from Qualcomm's portal.

### Option C: Request `GenieDialog_getQnnBackendHandle()` from Qualcomm

If Qualcomm adds a single C API function to `GenieDialog.h`:

```c
void* GenieDialog_getQnnBackend(GenieDialog_Handle_t dialog);
```

then no library rebuild is needed. The HybridDialog (or Rust code in genie-rs) calls this function after `GenieDialog_create()` to get the VTCM-configured backend handle, then loads MLP context binaries directly.

## 9. Summary

| Layer | What's Needed | Status |
|-------|---------------|--------|
| VTCM config | GCC 11.2 or matching ABI for extensions library | ❌ Only GCC 13.3 on Dragon |
| LibGenie rebuild | Full QAIRT SDK or Ubuntu 22.04 aarch64 | ❌ Not available |
| Backend handle sharing | `qnn_handles` globals injected into libGenie.so | ✅ Code written, needs rebuild |
| MLP context binaries | 36 × HTP binaries, no KV cache, INT8 | ✅ Built and deployed at `/tmp/mlp_htp2/` |
| HybridDialog | C++ class registered as GenieDialog type | ✅ Skeleton done at `genie-rs/src/hybrid/` |
| MLP execution via QNN C API | Raw VTable dispatch for `graphExecute` | ✅ Tested and working |
| Service running | 6 models on genie-rs | ✅ Verified via `/v1/chat/completions` |

### Root cause chain

```
Dragon runs OE Linux (custom Yocto distribution from Radxa)
  → g++-11 not available in its repositories
  → Cannot compile code with ABI matching the extensions library (GCC 11.2)
  → extensions library beforeBackendInitialize() returns 0 configs (silent ABI failure)
  → Cannot create VTCM-configured QNN backend
  → MLP context loading fails with error 5005
  → Hybrid pipeline (CPU attention + NPU MLP) remains blocked
```

All source code, MLP binaries, model weights, and integration infrastructure are ready. The single remaining dependency is a compiler ABI match with the proprietary `libQnnHtpNetRunExtensions.so`.
