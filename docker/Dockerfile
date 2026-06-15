# Multi-stage build for dragon-npu-ai hybrid pipeline
# Stage 1: Build libGenie.so with HybridDialog support
# Stage 2: Build MLP HTP context binaries
# Stage 3: Package for deployment

# ============================================================
# Stage 1: GenieDialog Hybrid Library
# ============================================================
FROM ubuntu:22.04 AS genie-builder

LABEL description="Dragon NPU AI Hybrid Pipeline Builder"
LABEL maintainer="danvincent"

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update -qq && apt-get install -y -qq \
    g++-11 \
    make \
    python3 \
    python3-pip \
    cargo \
    libc++-dev \
    libc++abi-dev \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up GCC 11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Install QAIRT SDK (mount at /opt/qairt)
# The SDK and repo are mounted at runtime

# Copy build script
COPY scripts/build-genie-hybrid.sh /build-genie-hybrid.sh
RUN chmod +x /build-genie-hybrid.sh

WORKDIR /opt/qairt

# Entry point: build script
CMD ["/build-genie-hybrid.sh"]

# ============================================================
# Stage 2: MLP HTP Context Binary Generator
# ============================================================
FROM genie-builder AS mlp-builder

# Install Python ML dependencies for model export
RUN pip3 install --quiet \
    torch \
    transformers \
    onnx \
    onnxscript \
    numpy \
    accelerate

# Copy the MLP exporter and conversion scripts
COPY scripts/export_all_mlp.py /export_all_mlp.py
COPY scripts/convert_mlp_int8.sh /convert_mlp_int8.sh

# The MLP pipeline runs in the mounted SDK context
# Model weights and QNN tools are accessed via /opt/qairt mount

# ============================================================
# Stage 3: Runtime (not used directly, shows required deps)
# ============================================================
FROM ubuntu:22.04 AS runtime

LABEL description="Dragon NPU AI Runtime Dependencies"

# Libraries needed on Dragon for genie-rs + hybrid pipeline
RUN apt-get update -qq && apt-get install -y -qq \
    libc++1 \
    libc++abi1 \
    libgomp1 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Usage
# ============================================================
# Build libGenie.so:
#   docker build -t dragon-npu-builder .
#   docker run --rm -it \
#     -v /path/to/qairt-sdk:/opt/qairt:Z \
#     -v /path/to/dragon-npu-api:/workspace:Z \
#     dragon-npu-builder
#
# Build MLP binaries:
#   docker run --rm -it \
#     -v /path/to/qairt-sdk:/opt/qairt:Z \
#     -v /path/to/dragon-npu-api:/workspace:Z \
#     --entrypoint /bin/bash \
#     dragon-npu-builder -c "python3 /export_all_mlp.py && bash /convert_mlp_int8.sh"
