# VTCM Unblock Strategy: Resolving C++ ABI Mismatch on QCS6490 V68

**Target Goal:** Bridge the GCC 11.2 / GCC 13.3 ABI gap to successfully initialize the QNN Htp Backend with correct Vector Tightly Coupled Memory (VTCM) allocations.

---

## Strategy 1: The Fedora 44 Cross-Compile Sandbox (Recommended)

Your x86_64 Fedora 44 server can act as a precise build environment by isolating a GCC 11 toolchain inside an OCI container. This bypasses host dependency updates and allows you to generate binaries fully compatible with Qualcomm's `libQnnHtpNetRunExtensions.so`.

### 1. Build Environment Setup
Ubuntu 22.04 LTS (Jammy) uses **GCC 11.4** as its base native/cross compiler, matching the C++ ABI layout (vtable ordering and `std::string` Small String Optimization) of Qualcomm’s toolchain.

Run the following on your Fedora 44 server:

```bash
# Spin up a persistent Ubuntu 22.04 workspace container
podman run -it \
  --name qairt-abi-builder \
  -v /path/to/your/sdk/and/code:/workspace:Z \
  ubuntu:22.04 \
  /bin/bash