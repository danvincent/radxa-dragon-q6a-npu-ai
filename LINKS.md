# External Resources

## Original Research

- **[Foadsf's NPU DMA fix gist](https://gist.github.com/Foadsf/...)** — **FIXME: Replace `...` with actual gist URL.** The original investigation that identified the `no reserved DMA memory for FASTRPC` issue on Qualcomm platforms. Search GitHub gists for "Foadsf fastrpc DMA".

## Model

- **[ModelScope: radxa/Llama3.2-1B-1024-qairt-v68](https://modelscope.cn/models/radxa/Llama3.2-1B-1024-qairt-v68)** — Pre-compiled QNN HTP context binary for Radxa Dragon Q6A. Download with `git lfs`.

## Qualcomm SDK

- **[QAIRT (Qualcomm AI Runtime) documentation](https://docs.qualcomm.com/doc/80-63442-10)** — Official docs for the Qualcomm AI Runtime toolkit.
- **[QAIRT ONNX to QNN Tutorial](https://docs.qualcomm.com/doc/80-63442-10/topic/onnx_to_qnn_tutorial_linux_host.html)** — Tutorial for converting ONNX models to QNN format.

## genie-rs

- **[genie-rs source repository](https://github.com/radxa-dragon/genie-rs)** — The Rust OpenAI-compatible API server that wraps Qualcomm's Genie SDK.

## Community

- **[Radxa Dragon Q6A documentation](https://radxa.com/products/accessories/dragon-q6a)** — Official Radxa product page.

## Tools

- **Device Tree Compiler**: `dtc`, `fdtoverlay` — provided by the `device-tree-compiler` package on Ubuntu.
- **bindgen**: Generates Rust FFI bindings from C headers. Used by genie-rs for Genie SDK integration.
