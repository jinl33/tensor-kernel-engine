#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace tke {
class ScopedTimer {
 public:
  explicit ScopedTimer(std::vector<double>& samples) : samples_(samples), start_(Clock::now()) {}
  ~ScopedTimer() {
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    samples_.push_back(elapsed);
  }
 private:
  using Clock = std::chrono::high_resolution_clock;
  std::vector<double>& samples_;
  Clock::time_point start_;
};

struct Percentiles {
  double p50;
  double p95;
  double p99;
};

inline Percentiles percentiles(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  auto at = [&](double q) { return samples[static_cast<size_t>(q * (samples.size() - 1))]; };
  return {at(0.50), at(0.95), at(0.99)};
}
}  // namespace tke
