#!/bin/bash
# Build GenieDialog with HybridDialog support inside Ubuntu 22.04 container.
# Run from /workspace (mounted dragon-npu-api repo)
set -euo pipefail

QAIRT=/opt/qairt
GENIE_DIR=$QAIRT/examples/Genie/Genie
SRC=$GENIE_DIR/src
WORKSPACE=/workspace
PATCH_DIR=$WORKSPACE/genie-rs/src/hybrid

echo "=== Setup ==="
apt-get update -qq && apt-get install -y -qq g++-11 make libc++-dev 2>&1 | tail -3
CXX=g++-11

echo "=== Apply HybridDialog patches ==="
# 1. Copy hybrid.hpp/cpp to dialogs dir
cp $PATCH_DIR/hybrid.hpp $SRC/qualla/dialogs/hybrid.hpp
cp $PATCH_DIR/hybrid.cpp $SRC/qualla/dialogs/hybrid.cpp

# 2. Copy qnn_handles.h/cpp to qnn-api dir
cp $PATCH_DIR/qnn_handles.h $SRC/qualla/engines/qnn-api/qnn_handles.h 2>/dev/null || true
cat > $SRC/qualla/engines/qnn-api/qnn_handles.h << 'EOF'
#ifndef QNN_HANDLES_H
#define QNN_HANDLES_H
extern "C" {
extern void* g_qnn_backend;
extern void* g_qnn_device;
}
#endif
EOF

cat > $SRC/qualla/engines/qnn-api/qnn_handles.cpp << 'EOF'
#include "qnn_handles.h"
extern "C" {
void* g_qnn_backend = nullptr;
void* g_qnn_device = nullptr;
}
EOF

# 3. Add HybridDialog::TYPE to Dialog::list()
grep -q "HybridDialog::TYPE" $SRC/qualla/dialog.cpp || \
sed -i '/SelfSpecDecDialog::TYPE/a\                                                      HybridDialog::TYPE,' $SRC/qualla/dialog.cpp

# 4. Add HybridDialog registration to Dialog::create()
grep -q "HybridDialog" $SRC/qualla/dialog.cpp | grep -q "create(" || \
sed -i '/if (type == SelfSpecDecDialog::TYPE)/i\    if (type == HybridDialog::TYPE) { return std::make_unique<HybridDialog>(env, name, conf); }' $SRC/qualla/dialog.cpp

# 5. Add hybrid to Dialog.cpp C API type validation
grep -q 'dialogType == "hybrid"' $SRC/Dialog.cpp || \
sed -i '/} else if (dialogType == "eaglet")/i\      } else if (dialogType == "hybrid") {\n      } else if (dialogType == "eaglet") {' $SRC/Dialog.cpp

# 6. Add include for hybrid.hpp in dialog.cpp
grep -q 'hybrid.hpp' $SRC/qualla/dialog.cpp || \
sed -i '/#include "dialogs\/ssd-q1.hpp"/a #include "dialogs/hybrid.hpp"' $SRC/qualla/dialog.cpp

echo "=== Patches applied ==="

echo "=== Build flags ==="
# Use QAIRT includes
INCLUDES="-I$GENIE_DIR/include -I$SRC -I$SRC/qualla/include -I$SRC/qualla -I$SRC/pipeline"
INCLUDES+=" -I$SRC/trace/include -I$SRC/quantization/include -I$SRC/resource-manager/include"
INCLUDES+=" -I$QAIRT/include -I$QAIRT/include/QNN -I$QAIRT/include/QNN/HTP -I$QAIRT/include/Genie"
INCLUDES+=" -I$SRC/qualla/engines/qnn-api -I$SRC/qualla/engines/qnn-api/config -I$SRC/qualla/engines/qnn-api/buffer"
INCLUDES+=" -I$SRC/qualla/engines/qnn-cpu -I$SRC/qualla/engines/qnn-gpu -I$SRC/qualla/engines/qnn-htp"
INCLUDES+=" -I$SRC/qualla/engines/qnn-htp/KVCache -I$SRC/qualla/engines/qnn-htp/nsp-utils"
INCLUDES+=" -I$SRC/qualla/adaptors -I$SRC/qualla/tokenizers -I$SRC/qualla/MmappedFile/include"
INCLUDES+=" -I$QAIRT/share/QNN/converter/jni"

CXXFLAGS="-std=c++2a -frtti -fPIC -O3 -Wno-write-strings -Wno-deprecated -fvisibility=hidden"
DEFS="-DGENIE_API=__attribute__((visibility(\"default\"))) -DSPILLFILL"
DEFS+=" -DQUALLA_ENGINE_QNN_CPU=TRUE -DQUALLA_ENGINE_QNN_GPU=TRUE"
DEFS+=" -DFMT_HEADER_ONLY -DGENIE_SAMPLE -DQUALLA_ENGINE_QNN_HTP=TRUE"

OBJ=/tmp/genie_build
mkdir -p $OBJ

echo "=== Compiling sources ==="
N=0
cc() { N=$((N+1)); local s=$1 o=$2; printf "\r  [%02d] %-40s" $N "$o"; $CXX -c $CXXFLAGS $DEFS $INCLUDES "$s" -o "$OBJ/$o.o" 2>/dev/null || { echo "FAIL $o"; return 1; }; }

# Core
for f in dialog context sampler tokenizer engine env encoder; do
    cc "$SRC/qualla/$f.cpp" "$f"
done

# Pipeline
for f in "$SRC/pipeline/"*.cpp; do [ -f "$f" ] && cc "$f" "pipeline_$(basename ${f%.cpp})"; done

# Trace
for f in "$SRC/qualla/trace/src/"*.cpp; do [ -f "$f" ] && cc "$f" "trace_$(basename ${f%.cpp})"; done

# Quantization (skip if dir empty)
for f in "$SRC/qualla/quantization/src/"*.cpp; do [ -f "$f" ] && cc "$f" "quant_$(basename ${f%.cpp})"; done

# Resource Manager
for f in "$SRC/qualla/resource-manager/src/"*.cpp; do [ -f "$f" ] && cc "$f" "rm_$(basename ${f%.cpp})"; done

# QNN API
for f in "$SRC/qualla/engines/qnn-api/"*.cpp; do
    [ -f "$f" ] && cc "$f" "qnn_$(basename ${f%.cpp})"
done
for f in "$SRC/qualla/engines/qnn-api/config/"*.cpp; do [ -f "$f" ] && cc "$f" "cfg_$(basename ${f%.cpp})"; done
for f in "$SRC/qualla/engines/qnn-api/buffer/"*.cpp; do
    [ -f "$f" ] && ! echo "$f" | grep -qi "Dx\|Dma" && cc "$f" "buf_$(basename ${f%.cpp})"
done
for f in "$SRC/qualla/engines/qnn-api/buffer/Allocator/"*.cpp; do
    [ -f "$f" ] && ! echo "$f" | grep -qi "dx\|dma" && cc "$f" "alloc_$(basename ${f%.cpp})"
done
for f in "$SRC/qualla/engines/qnn-api/buffer/Registration/"*.cpp; do
    [ -f "$f" ] && ! echo "$f" | grep -qi "dx\|dma" && cc "$f" "reg_$(basename ${f%.cpp})"
done
for f in "$SRC/qualla/engines/qnn-api/PAL/linux/"*.cpp; do [ -f "$f" ] && cc "$f" "pal_$(basename ${f%.cpp})"; done

# Engine state
for f in "$SRC/qualla/engine-state/"*.cpp; do [ -f "$f" ] && cc "$f" "state_$(basename ${f%.cpp})"; done

# Engines
cc "$SRC/qualla/engines/qnn-htp.cpp" qnn_htp
for f in "$SRC/qualla/engines/qnn-cpu/"*.cpp; do [ -f "$f" ] && cc "$f" "cpu_$(basename ${f%.cpp})"; done
for f in "$SRC/qualla/engines/qnn-gpu/"*.cpp; do [ -f "$f" ] && cc "$f" "gpu_$(basename ${f%.cpp})"; done

# Dialogs
for f in "$SRC/qualla/dialogs/"*.cpp; do [ -f "$f" ] && cc "$f" "dlg_$(basename ${f%.cpp})"; done

# Top-level C API files
for f in "$SRC/"*.cpp; do
    bn=$(basename "$f")
    # Skip files that need DX/DMA or platform-specific deps
    case "$bn" in
        Accuracy.cpp|Dlc.cpp|Embedding.cpp|EncoderDecoder.cpp) continue ;;
        GenieDialog.cpp|GenieDialogEmbedding.cpp|GenieCommon.cpp|GenieEngine.cpp) ;;
        GenieLog.cpp|GenieNode.cpp|GeniePipeline.cpp|GenieProfile.cpp) ;;
        GenieSampler.cpp|GenieTokenizer.cpp|GenieAccuracy.cpp|GenieDlc.cpp) ;;
        GenieEmbedding.cpp) ;;
        Context.cpp|Engine.cpp|Logger.cpp|LogUtils.cpp|PlatformDetector.Default.cpp) ;;
        Profile.cpp|Registry.cpp|Sampler.cpp|Tokenizer.cpp|Util.cpp|Dialog.cpp) ;;
        *) continue ;;
    esac
    cc "$f" "top_$(basename ${f%.cpp})"
done

echo ""
echo "=== Compilation complete ($N files) ==="

echo "=== Linking ==="
OUTDIR=$QAIRT/lib/aarch64-linux-gnu
mkdir -p $OUTDIR
$CXX -shared -s -fPIC -o $OUTDIR/libGenie.so $OBJ/*.o \
    -L$SRC/qualla/tokenizers/rust/target/release -ltokenizers_capi -ldl -pthread 2>&1
echo "Link: $?"
ls -lh $OUTDIR/libGenie.so

# Verify C API exported
echo "=== C API symbols ==="
nm -D $OUTDIR/libGenie.so | grep "GenieDialogConfig_createFromJson" | head -2
echo ""
echo "=== Hybrid type registered ==="
nm -C $OUTDIR/libGenie.so | grep "Dialog::list" | head -2

echo ""
echo "=== DONE ==="
echo "Library at: $OUTDIR/libGenie.so"
echo "Ship to Dragon: scp $OUTDIR/libGenie.so daniel@dragon:\$HOME/qairt/2.47.0.260601/lib/aarch64-oe-linux-gcc11.2/"
