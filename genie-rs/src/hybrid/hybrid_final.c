#include <dlfcn.h>
#include <stdio.h>
#include <string>
#include "Genie/GenieDialog.h"
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnLog.h"
#include "QnnDevice.h"

namespace qnn_tools_netrun {
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual bool initialize(void* lib) = 0;
    virtual bool loadConfig(std::string f) = 0;
    virtual bool setOffloadGraphs(bool e) = 0;
    virtual bool beforeBackendInit(QnnBackend_Config_t*** c, uint32_t* n) = 0;
    virtual bool afterBackendInit() = 0;
    virtual bool prepareSoc(uint32_t d, const std::string& a, uint32_t v, const std::string& n, uint32_t o, uint32_t l, uint32_t r, uint32_t h) = 0;
    virtual bool setGraphPriority(uint32_t, Qnn_Priority_t) = 0;
    virtual bool setupLogging(QnnLog_Callback_t cb, QnnLog_Level_t l) = 0;
    virtual bool setPerfProfile(int) = 0;
    virtual bool setParentAppType(int) = 0;
};
typedef IBackend* (*CreateFn)();
}

void log_cb(const char* m, QnnLog_Level_t l, uint64_t t, va_list a) {}

typedef Qnn_ErrorHandle_t (*BC)(Qnn_LogHandle_t, const QnnBackend_Config_t**, Qnn_BackendHandle_t*);

int main(int argc, char** argv) {
    const char* mlp = argc > 1 ? argv[1] : "/tmp/mlp_htp2/qwen_mlp_00.QCS6490.bin";
    
    // 1. Init GenieDialog
    const char* cfg = "{\"dialog\":{\"version\":1,\"type\":\"basic\",\"max-num-tokens\":8,\"context\":{\"version\":1,\"size\":4096,\"n-vocab\":128256,\"bos-token\":128000,\"eos-token\":128009,\"pad-token\":128004},\"sampler\":{\"version\":1,\"seed\":42,\"temp\":0.8,\"top-k\":1,\"top-p\":0.95},\"tokenizer\":{\"version\":1,\"path\":\"/home/daniel/llama-4096-v68-model/tokenizer.json\"},\"engine\":{\"version\":1,\"n-threads\":3,\"backend\":{\"version\":1,\"type\":\"QnnHtp\",\"QnnHtp\":{\"version\":1,\"use-mmap\":true,\"spill-fill-bufsize\":0,\"mmap-budget\":0,\"poll\":true,\"cpu-mask\":\"0xe0\",\"kv-dim\":64,\"pos-id-dim\":32,\"allow-async-init\":true},\"extensions\":\"/home/daniel/llama-4096-v68-model/htp_backend_ext_config.json\"},\"model\":{\"version\":1,\"type\":\"binary\",\"binary\":{\"version\":1,\"ctx-bins\":[\"/home/daniel/llama-4096-v68-model/models/weight_sharing_model_1_of_1.serialized.bin\"]}}}}}";
    
    printf("1. GenieDialog init...\n");
    GenieDialogConfig_Handle_t gc = NULL; GenieDialogConfig_createFromJson(cfg, &gc);
    GenieDialog_Handle_t gd = NULL;
    if (GenieDialog_create(gc, &gd) != GENIE_STATUS_SUCCESS) { printf("FAIL\n"); return 1; }
    printf("OK\n");
    
    // 2. Get QNN API
    void* qnn = dlopen("/home/daniel/qairt/2.47.0.260601/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so", RTLD_NOW);
    auto gp = (uint32_t (*)(const QnnInterface_t***, uint32_t*))dlsym(qnn, "QnnInterface_getProviders");
    const QnnInterface_t** p = NULL; uint32_t n = 0; gp(&p, &n);
    const auto* api = &p[0]->v2_36;
    printf("2. QNN API OK (%u providers)\n", n);
    
    // 3. Init extensions
    void* ext_lib = dlopen("libQnnHtpNetRunExtensions.so", RTLD_NOW | RTLD_LOCAL);
    auto* ext = ((qnn_tools_netrun::CreateFn)dlsym(ext_lib, "createBackendInterface"))();
    printf("3. Extensions: %p\n", (void*)ext);
    
    // 4. Initialize
    printf("4. Check symbols in qnn handle: %s\n", dlsym(qnn, "QnnInterface_getProviders") ? "FOUND" : "MISSING");
    if (!ext->initialize(qnn)) { printf("   init FAIL\n"); return 1; }
    printf("   init OK\n");
    
    // 5. Setup SOC
    ext->setupLogging(log_cb, QNN_LOG_LEVEL_ERROR);
    if (!ext->prepareSoc(0, "v68", 4, "qcs6490", 0, 0, 0, 0)) { printf("   prepareSoc FAIL\n"); }
    else printf("   prepareSoc OK\n");
    
    // 6. Get configs
    QnnBackend_Config_t** bcfg = NULL; uint32_t bcn = 0;
    ext->beforeBackendInit(&bcfg, &bcn);
    printf("6. Configs: %u\n", bcn);
    for (uint32_t i = 0; i < bcn; i++) printf("   cfg[%u].option=%u\n", i, bcfg[i]->option);
    
    // 7. Create backend
    Qnn_LogHandle_t lh = NULL; api->logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &lh);
    Qnn_BackendHandle_t bh = NULL;
    ((BC)api->backendCreate)(lh, (const QnnBackend_Config_t**)bcfg, &bh);
    printf("7. Backend: %s\n", bh ? "OK" : "FAIL");
    if (!bh) return 1;
    ext->afterBackendInit();
    
    Qnn_DeviceHandle_t dh = NULL; api->deviceCreate(lh, NULL, &dh);
    
    // 8. Load MLP
    FILE* f = fopen(mlp, "rb");
    fseek(f,0,SEEK_END); uint64_t sz = ftell(f); fseek(f,0,SEEK_SET);
    void* d = malloc((size_t)sz); fread(d,1,sz,f); fclose(f);
    
    Qnn_ContextHandle_t cx = NULL;
    uint32_t e = api->contextCreateFromBinary(bh, dh, NULL, d, sz, &cx, NULL);
    free(d);
    
    if (e == QNN_SUCCESS) {
        printf("8. *** MLP CONTEXT OK! ***\n");
        Qnn_GraphHandle_t gr = NULL;
        api->graphRetrieve(cx, "qwen_mlp_00", &gr);
        if (gr) {
            float in[2048]={0}, out[2048]={0};
            uint32_t dm[3]={1,1,2048};
            Qnn_Tensor_t it=QNN_TENSOR_INIT, ot=QNN_TENSOR_INIT;
            it.version=2; it.v2.name="mlp_input"; it.v2.type=3; it.v2.rank=3; it.v2.dimensions=dm; it.v2.clientBuf.data=in; it.v2.clientBuf.dataSize=sizeof(in);
            ot.version=2; ot.v2.name="mlp_output"; ot.v2.type=2; ot.v2.rank=3; ot.v2.dimensions=dm; ot.v2.clientBuf.data=out; ot.v2.clientBuf.dataSize=sizeof(out);
            if (api->graphExecute(gr, &it, 1, &ot, 1, NULL, NULL) == QNN_SUCCESS)
                printf("   EXECUTE OK out[0..3]: %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
            else printf("   EXECUTE FAIL\n");
            api->contextFree(cx, NULL);
        }
    } else printf("8. MLP FAILED (err=%u)\n", (unsigned)e);
    
    if (dh) api->deviceFree(dh); api->backendFree(bh); if (lh) api->logFree(lh);
    GenieDialog_free(gd); GenieDialogConfig_free(gc);
    return 0;
}
