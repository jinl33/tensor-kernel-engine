#pragma once

#include <cstddef>
#include <string>

namespace tke {
class InferenceEngine {
 public:
  explicit InferenceEngine(std::string model_path = {});
  void warmup(std::size_t iterations = 1000);
  double run_once();
 private:
  std::string model_path_;
};
}  // namespace tke
