//==============================================================================
//
//  Hybrid Dialog: CPU attention + NPU MLP for unlimited context.
//
//==============================================================================
#include <dlfcn.h>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdio>

#include "Trace.hpp"
#include "hybrid.hpp"
#include "qualla/detail/Log.hpp"

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnLog.h"
#include "QnnDevice.h"

namespace qualla {

// ============================================================
// Safetensors loader (reads BF16/F16/F32 weights)
// ============================================================
struct STensor {
    std::vector<uint64_t> shape;
    std::string dtype;
    uint64_t offset;
    uint64_t size;
};

static std::unordered_map<std::string, STensor> load_safetensors_index(const std::string& path) {
    std::ifstream f(path);
    std::unordered_map<std::string, STensor> result;
    if (!f) return result;
    auto json = nlohmann::json::parse(f);
    
    // Determine base directory from index path
    auto base = path.substr(0, path.find_last_of("/\\") + 1);
    
    // Get weight map
    auto& wm = json["weight_map"];
    std::unordered_map<std::string, std::string> file_for_name;
    for (auto& [name, file] : wm.items())
        file_for_name[name] = base + file.get<std::string>();
    
    // Store file mapping for later access
    // (We'll just store by name, load on demand)
    for (auto& [name, file] : file_for_name) {
        STensor st;
        st.shape = {};  // Will be filled when loaded
        result[name] = st;
    }
    result["_file_map"] = STensor{};
    for (auto& [name, file] : file_for_name)
        result["_file_map"].shape.push_back(0);  // placeholder
    return result;
}

// Simple FP16 -> FP32 converter
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 31;
    uint32_t mant = h & 1023;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (0x7f - 15 << 23) | (mant << 13);
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

// ============================================================
// Math helpers
// ============================================================
static float dot(const float* a, const float* b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void matvec(float* out, const float* vec, const float* mat, int m, int n) {
    // out[m] = vec[n] * mat[n][m]
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++)
            sum += vec[j] * mat[j * m + i];
        out[i] = sum;
    }
}

static void rmsnorm(float* out, const float* x, const float* w, int d, float eps) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / d + eps);
    for (int i = 0; i < d; i++) out[i] = w[i] * (x[i] * r);
}

static void rope(float* q, float* k, int head_dim, int pos, const float* cos_pre, const float* sin_pre) {
    for (int j = 0; j < head_dim; j += 2) {
        float c = cos_pre[pos * (head_dim/2) + j/2];
        float s = sin_pre[pos * (head_dim/2) + j/2];
        float q0 = q[j], q1 = q[j+1];
        q[j] = q0 * c - q1 * s;
        q[j+1] = q0 * s + q1 * c;
        float k0 = k[j], k1 = k[j+1];
        k[j] = k0 * c - k1 * s;
        k[j+1] = k0 * s + k1 * c;
    }
}

// ============================================================
// HybridDialog
// ============================================================
HybridDialog::HybridDialog(std::shared_ptr<Env> env,
                           const std::string& name,
                           const nlohmann::json& conf)
    : Dialog(env, name, conf) {
    // Store config for later init
    if (conf.contains("hidden-dim")) m_hidden_dim = conf["hidden-dim"].get<int>();
    if (conf.contains("num-layers")) m_num_layers = conf["num-layers"].get<int>();
    if (conf.contains("num-heads")) m_num_heads = conf["num-heads"].get<int>();
    if (conf.contains("num-kv-heads")) m_num_kv_heads = conf["num-kv-heads"].get<int>();
    if (conf.contains("mlp-bin-dir")) m_mlp_dir = conf["mlp-bin-dir"].get<std::string>();
    if (conf.contains("weights-dir")) m_weights_dir = conf["weights-dir"].get<std::string>();
    m_head_dim = m_hidden_dim / m_num_heads;
    completeInit();
}

HybridDialog::~HybridDialog() {
    if (m_qnn_api) {
        auto* api = (const QnnInterface_ImplementationV2_36_t*)m_qnn_api;
        for (auto* ctx : m_mlp_ctxs)
            if (ctx) api->contextFree((Qnn_ContextHandle_t)ctx, NULL);
    }
}

void HybridDialog::completeInit() {
    if (m_initFinished) return;
    Dialog::completeInit();

    QNN_INFO("HybridDialog: %d layers, %d hidden, %d heads, %d KV, %s",
             m_num_layers, m_hidden_dim, m_num_heads, m_num_kv_heads, m_mlp_dir.c_str());

    // Allocate KV cache (CPU managed, up to context_size)
    int max_seq = _ctx ? (int)_ctx->size() : 8192;
    int kv_dim = m_num_kv_heads * m_head_dim;
    m_k_cache.resize(m_num_layers, std::vector<float>(max_seq * kv_dim, 0));
    m_v_cache.resize(m_num_layers, std::vector<float>(max_seq * kv_dim, 0));

    // Init QNN and load MLP bins
    initQnnMlp();
    loadMlpBins();

    // Precompute RoPE
    // (done on first process call with actual max_seq)
}

bool HybridDialog::initQnnMlp() {
    m_qnn_backend = g_qnn_backend;
    m_qnn_device = g_qnn_device;

    if (!m_qnn_backend) {
        QNN_ERROR("QNN backend not initialized. Loading directly...");
        void* lib = dlopen("libQnnHtp.so", RTLD_NOLOAD | RTLD_GLOBAL);
        if (!lib) lib = dlopen("/home/daniel/qairt/2.47.0.260601/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so",
                              RTLD_NOW | RTLD_GLOBAL);
        if (!lib) { QNN_ERROR("Cannot load libQnnHtp.so"); return false; }

        auto get_p = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
            dlsym(lib, "QnnInterface_getProviders");
        if (!get_p) return false;

        const QnnInterface_t** providers = nullptr;
        uint32_t num = 0;
        if (get_p(&providers, &num) != QNN_SUCCESS || num == 0) return false;
        m_qnn_api = &providers[0]->v2_36;
        auto* api = (const QnnInterface_ImplementationV2_36_t*)m_qnn_api;

        Qnn_LogHandle_t log = nullptr;
        api->logCreate([](const char*, QnnLog_Level_t, uint64_t, va_list){}, QNN_LOG_LEVEL_ERROR, &log);

        typedef Qnn_ErrorHandle_t (*BCFn)(Qnn_LogHandle_t, const QnnBackend_Config_t**, Qnn_BackendHandle_t*);
        Qnn_BackendHandle_t bh = nullptr;
        ((BCFn)api->backendCreate)(log, nullptr, &bh);
        m_qnn_backend = bh;

        Qnn_DeviceHandle_t dh = nullptr;
        api->deviceCreate(log, nullptr, &dh);
        m_qnn_device = dh;
    } else {
        // Use shared globals
        auto* api = (const QnnInterface_ImplementationV2_36_t*)m_qnn_api;
    }
    return m_qnn_backend != nullptr;
}

bool HybridDialog::loadMlpBins() {
    auto* api = (const QnnInterface_ImplementationV2_36_t*)m_qnn_api;
    if (!api) return false;

    for (int i = 0; i < m_num_layers; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/qwen_mlp_%02d.QCS6490.bin", m_mlp_dir.c_str(), i);

        FILE* f = fopen(path, "rb");
        if (!f) { QNN_WARN("MLP %d not found: %s", i, path); continue; }
        fseek(f, 0, SEEK_END);
        uint64_t sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        void* data = malloc(sz);
        fread(data, 1, sz, f);
        fclose(f);

        Qnn_ContextHandle_t ctx = nullptr;
        uint32_t err = api->contextCreateFromBinary(
            (Qnn_BackendHandle_t)m_qnn_backend, (Qnn_DeviceHandle_t)m_qnn_device,
            NULL, data, sz, &ctx, NULL);
        free(data);

        if (err != QNN_SUCCESS) {
            QNN_WARN("MLP %d ctx FAILED (err=%u)", i, (unsigned)err);
            m_mlp_ctxs.push_back(nullptr);
            m_mlp_graphs.push_back(nullptr);
            continue;
        }
        char gname[32];
        snprintf(gname, sizeof(gname), "qwen_mlp_%02d", i);
        Qnn_GraphHandle_t graph = nullptr;
        if (api->graphRetrieve(ctx, gname, &graph) != QNN_SUCCESS || !graph) {
            QNN_WARN("MLP %d no graph", i);
            api->contextFree(ctx, NULL);
            m_mlp_ctxs.push_back(nullptr);
            m_mlp_graphs.push_back(nullptr);
            continue;
        }
        m_mlp_ctxs.push_back(ctx);
        m_mlp_graphs.push_back(graph);
    }
    QNN_INFO("MLP: %zu/%d loaded", m_mlp_ctxs.size(), m_num_layers);
    return m_mlp_ctxs.size() > 0;
}

bool HybridDialog::executeMlp(int layer, const float* input, float* output) {
    if ((size_t)layer >= m_mlp_ctxs.size() || !m_mlp_graphs[layer]) {
        memcpy(output, input, m_hidden_dim * sizeof(float));
        return true;
    }
    auto* api = (const QnnInterface_ImplementationV2_36_t*)m_qnn_api;
    auto graph = (Qnn_GraphHandle_t)m_mlp_graphs[layer];
    uint32_t dims[3] = {1, 1, (uint32_t)m_hidden_dim};

    Qnn_Tensor_t in_t = QNN_TENSOR_INIT;
    in_t.version = QNN_TENSOR_VERSION_2;
    in_t.v2.name = "mlp_input";
    in_t.v2.type = QNN_TENSOR_TYPE_APP_WRITE;
    in_t.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    in_t.v2.dataType = QNN_DATATYPE_FLOAT_32;
    in_t.v2.rank = 3; in_t.v2.dimensions = dims;
    in_t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    in_t.v2.clientBuf.data = (void*)input;
    in_t.v2.clientBuf.dataSize = m_hidden_dim * sizeof(float);

    Qnn_Tensor_t out_t = QNN_TENSOR_INIT;
    out_t.version = QNN_TENSOR_VERSION_2;
    out_t.v2.name = "mlp_output"; out_t.v2.type = QNN_TENSOR_TYPE_APP_READ;
    out_t.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    out_t.v2.dataType = QNN_DATATYPE_FLOAT_32;
    out_t.v2.rank = 3; out_t.v2.dimensions = dims;
    out_t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    out_t.v2.clientBuf.data = output;
    out_t.v2.clientBuf.dataSize = m_hidden_dim * sizeof(float);

    uint32_t err = api->graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL);
    if (err != QNN_SUCCESS) {
        QNN_ERROR("MLP %d exec FAILED (err=%u)", layer, (unsigned)err);
        memcpy(output, input, m_hidden_dim * sizeof(float));
        return false;
    }
    return true;
}

void HybridDialog::cpuLayer(int layer, float* hidden, int pos,
                            const float* cos, const float* sin) {
    // Placeholder: for now, this copies hidden through.
    // Full implementation requires attention weights loaded from safetensors.
    // The MLP is called by the caller after this returns.
}

bool HybridDialog::process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) {
    GENIE_TRACE();

    auto& engine = *_engine["primary"];
    auto& sampler = *_sampler["primary"];

    // Use standard engine for token processing (embedding, etc.)
    Tensor logits; // Tensor is a typedef in Genie
    if (engine.process(tokens, logits, false) != 1) {
        return Dialog::abort("Engine failed", callback);
    }

    // Generation loop
    while (true) {
        if (State::canceled()) break;
        if (_n_past + 1 > _ctx->size()) {
            // Context limit reached - for hybrid this is CPU-RAM limited
            // We can extend the KV cache, but sliding window would start here
            QNN_WARN("Context limit: %zu", _ctx->size());
        }

        // Sample next token using GenieDialog's sampler
        tokens[0] = _last_tok = sampler.process(logits);
        sampler.updateSampledTokenHistory(tokens[0]);
        _n_past++; _n_generated++; _n_decode++;
        engine.updateTokenCheckpoint((uint32_t)_last_tok, _n_past);
        engine.updateKV(_n_past);

        if (_ctx->is_eos(_last_tok) || !callback.callBack(tokens.data(), 1, Sentence::CONTINUE, tokenizer()))
            break;
    }
    callback.callBack(nullptr, 0, Sentence::END, tokenizer());
    return true;
}

bool HybridDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
    qualla::DialogCallback cb;
    cb.setCallBackType(qualla::QUALLA_CALLBACK_TYPE_TEXT);
    cb.getQueryCbFunc() = std::make_shared<std::function<bool(const std::string&, qualla::Sentence::Code)>>();
    *cb.getQueryCbFunc() = callback;
    return process(tokens, cb);
}

// Required overrides
bool HybridDialog::process(std::vector<int32_t>& tokens,
                           std::vector<size_t>& tokenNumPerBatch,
                           qualla::DialogCallback callback) {
    return process(tokens, callback);
}
bool HybridDialog::process(std::vector<uint8_t>&, Dialog::T2ECallback, Dialog::Callback) { return false; }
bool HybridDialog::process(std::vector<uint8_t>&, Dialog::T2ECallback, qualla::DialogCallback) { return false; }
bool HybridDialog::process(std::vector<int32_t>&, std::vector<size_t>&, Dialog::BatchCallback) { return false; }

} // namespace qualla
