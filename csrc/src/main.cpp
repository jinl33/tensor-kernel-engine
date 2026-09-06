#include <iostream>
#include <vector>

#include "engine.hpp"
#include "timer.hpp"

int main() {
  tke::InferenceEngine engine;
  engine.warmup();
  std::vector<double> samples;
  for (int i = 0; i < 1000; ++i) {
    tke::ScopedTimer timer(samples);
    engine.run_once();
  }
  const auto stats = tke::percentiles(samples);
  std::cout << "kernel_bench p50=" << stats.p50 << " ms p95=" << stats.p95
            << " ms p99=" << stats.p99 << " ms\n";
  return 0;
}
