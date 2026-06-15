/* HybridDialog wrapper library. LD_PRELOAD this to add HybridDialog type.
 * Compile on Dragon:
 *   g++ -shared -fPIC -std=c++17 -o libhybrid.so hybrid_wrapper.cpp \
 *       -I$SDK/include/QNN -I$SDK/include/Genie \
 *       -L$SDK/lib/aarch64-oe-linux-gcc11.2 -lGenie -ldl -lm
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

/* GenieDialog API */
#include "Genie/GenieDialog.h"

/* QNN API for MLP execution */
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnLog.h"
#include "QnnDevice.h"

/* ======================== QNN Backend Handle Capture ======================== */
static void* g_qnn_lib = nullptr;
static const QnnInterface_ImplementationV2_36_t* g_qnn_api = nullptr;
static Qnn_BackendHandle_t g_qnn_backend = nullptr;
static Qnn_DeviceHandle_t g_qnn_device = nullptr;
static Qnn_LogHandle_t g_qnn_log = nullptr;

/* Intercept dlopen to capture libQnnHtp handle */
extern "C" void* dlopen(const char* file, int mode) {
    static void* (*real_dlopen)(const char*, int) = nullptr;
    if (!real_dlopen) real_dlopen = (void*(*)(const char*,int))dlsym(RTLD_NEXT, "dlopen");
    void* handle = real_dlopen(file, mode);
    if (handle && file) {
        if (strstr(file, "libQnnHtp.so") && !strstr(file, "V68") && !strstr(file, "NetRun")) {
            g_qnn_lib = handle;
            fprintf(stderr, "[HYBRID] Captured QNN lib: %p\n", handle);
            // Get QNN API
            auto gp = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
                dlsym(handle, "QnnInterface_getProviders");
            if (gp) {
                const QnnInterface_t** p = nullptr; uint32_t n = 0;
                gp(&p, &n);
                if (n > 0) {
                    g_qnn_api = &p[0]->v2_36;
                    fprintf(stderr, "[HYBRID] QNN API version 2.36\n");
                }
            }
        }
    }
    return handle;
}

/* ======================== Original function pointers ======================== */
typedef std::vector<std::string> (*ListFn)();
typedef std::unique_ptr<class qualla::Dialog> (*CreateFn)(std::shared_ptr<class qualla::Env>, const std::string&, const nlohmann::json&);
typedef Genie_Status_t (*ConfigCreateFn)(const char*, GenieDialogConfig_Handle_t*);

static ListFn real_list = nullptr;
static CreateFn real_create = nullptr;

/* ======================== Intercept Dialog::list() ======================== */
/* We need to find the mangled symbol name for Dialog::list() */
/* In the pre-compiled lib: _ZN6qualla6Dialog4listEv */

extern "C" std::vector<std::string> _ZN6qualla6Dialog4listEv() {
    if (!real_list) real_list = (ListFn)dlsym(RTLD_NEXT, "_ZN6qualla6Dialog4listEv");
    auto types = real_list ? real_list() : std::vector<std::string>();
    types.push_back("hybrid");
    return types;
}

/* ======================== Intercept Dialog::create() ======================== */
/* Mangled: _ZN6qualla6Dialog6createENSt3__110shared_ptrINS_3EnvEEERKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEERKN9nlohmann3json3v3_12_05basic_jsonINS9_7details9adl_serializerENS3_6vectorIjNS3_9allocatorIjEEEEN3_3_0ENS3_6vectorIhNSD_IhEEEENS3_12basic_stringIcNS3_11char_traitsIcEENSD_IcEEEEEiNS3_7moneypunctIcNS3_10money_baseEEEEEvE */
/* This is very long. Let me use dlsym instead of a direct definition */

/* ======================== Initialize QNN backend for MLP ======================== */
// Called when GenieDialog creates the backend

extern "C" void init_mlp() {
    if (g_qnn_api && !g_qnn_backend) {
        fprintf(stderr, "[HYBRID] Initializing MLP...\n");
        
        // Create backend (same as GenieDialog does)
        typedef Qnn_ErrorHandle_t (*BCFn)(Qnn_LogHandle_t, const QnnBackend_Config_t**, Qnn_BackendHandle_t*);
        
        g_qnn_api->logCreate([](const char*, QnnLog_Level_t, uint64_t, va_list){}, QNN_LOG_LEVEL_ERROR, &g_qnn_log);
        
        if (g_qnn_api->backendCreate) {
            Qnn_BackendHandle_t bh = nullptr;
            if (((BCFn)g_qnn_api->backendCreate)(g_qnn_log, nullptr, &bh) == 0 && bh) {
                g_qnn_backend = bh;
                fprintf(stderr, "[HYBRID] QNN backend: %p\n", (void*)bh);
            }
        }
        if (g_qnn_api->deviceCreate) {
            g_qnn_api->deviceCreate(g_qnn_log, nullptr, &g_qnn_device);
        }
    }
}

/* ======================== Constructor ======================== */
__attribute__((constructor)) void init() {
    fprintf(stderr, "[HYBRID] HybridDialog wrapper loaded\n");
}
