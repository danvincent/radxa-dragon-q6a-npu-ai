//==============================================================================
//
//  Hybrid Dialog: CPU attention + NPU MLP for unlimited context.
//
//==============================================================================
#pragma once
#include "qualla/dialog.hpp"
#include <vector>
#include <string>

namespace qualla {

class HybridDialog : public Dialog {
 public:
  static constexpr const char* TYPE = "hybrid";

  HybridDialog(std::shared_ptr<Env> env, const std::string& name, const nlohmann::json& conf);
  virtual ~HybridDialog();

  virtual bool process(std::vector<int32_t>& tokens,
                       Dialog::Callback callback) override;
  virtual bool process(std::vector<int32_t>& tokens,
                       qualla::DialogCallback callback) override;
  virtual bool process(std::vector<int32_t>& tokens,
                       std::vector<size_t>& tokenNumPerBatch,
                       qualla::DialogCallback callback) override;
  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;
  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       qualla::DialogCallback callback) override;
  virtual bool process(std::vector<int32_t>& tokens,
                       std::vector<size_t>& tokenNumPerBatch,
                       Dialog::BatchCallback callback) override;
  virtual bool supportsPauseResume() override { return false; }
  void completeInit() override;
  virtual const char* getTraceNamespace() const override { return "Dialog::Hybrid"; }

 protected:
  virtual bool supportsLongContext() const override { return true; }

 private:
  // MLP context binary handles
  std::vector<void*> m_mlp_ctxs;
  std::vector<void*> m_mlp_graphs;

  // Model config
  int m_hidden_dim = 2048;
  int m_num_layers = 36;
  int m_num_heads = 16;
  int m_num_kv_heads = 2;
  int m_head_dim = 128;

  // Paths
  std::string m_mlp_dir = "/tmp/mlp_htp2";
  std::string m_weights_dir;

  // QNN handles
  void* m_qnn_backend = nullptr;
  void* m_qnn_device = nullptr;
  const void* m_qnn_api = nullptr;

  // KV cache (CPU managed, unlimited)
  std::vector<std::vector<float>> m_k_cache;
  std::vector<std::vector<float>> m_v_cache;

  bool initQnnMlp();
  bool loadMlpBins();
  bool executeMlp(int layer, const float* input, float* output);
};

} // namespace qualla
