#include "engine.hpp"

#include <chrono>
#include <thread>
#include <utility>

namespace tke {
InferenceEngine::InferenceEngine(std::string model_path) : model_path_(std::move(model_path)) {}
void InferenceEngine::warmup(std::size_t iterations) {
  for (std::size_t i = 0; i < iterations; ++i) run_once();
}
double InferenceEngine::run_once() {
  const auto start = std::chrono::high_resolution_clock::now();
  std::this_thread::yield();
  return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
}
}  // namespace tke
