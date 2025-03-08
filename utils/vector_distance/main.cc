#include <chrono>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include "distance.h"

namespace {

auto GenerateRandomVector(size_t size) -> std::vector<Float> {
  std::vector<Float> result(size);
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    result[i] = static_cast<Float>(rand()) / RAND_MAX;
  }
  return result;
}

// Benchmark L2Sq distance, compare time with scalar, AVX2 and AVX512F
auto Benchmark(size_t dim, size_t num_iters) {
  std::vector<Float> a = GenerateRandomVector(dim);
  std::vector<Float> b = GenerateRandomVector(dim);

  Float result_scalar = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_iters; ++i) {
    result_scalar += rox::GetDistanceL2SqScalar(a, b);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto scalar_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  Float result_avx2 = 0.0;
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_iters; ++i) {
    result_avx2 += rox::GetDistanceL2SqAvx2(a, b);
  }
  end = std::chrono::high_resolution_clock::now();
  auto avx2_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  auto diff_avx2 = std::abs(result_scalar - result_avx2);

  Float result_avx512f = 0.0;
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_iters; ++i) {
    result_avx512f += rox::GetDistanceL2SqAvx512F(a, b);
  }
  end = std::chrono::high_resolution_clock::now();
  auto avx512f_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  auto diff_avx512f = std::abs(result_scalar - result_avx512f);

  return std::make_tuple(scalar_time, avx2_time, avx512f_time, diff_avx2,
                         diff_avx512f);
}

}  // namespace

int main() {
  const std::vector<size_t> dims = {128, 256, 512, 1024, 2048, 4096};
  const size_t num_iters = 1000000;  // 1M
  const size_t rounds = 10;

  std::ofstream csv_file("benchmark_results.csv");
  // Write CSV header
  csv_file
      << "round,dim,scalar_time,avx2_time,avx512_time,diff_avx2,diff_avx512"
      << std::endl;

  // Warmup
  auto _ = Benchmark(128, num_iters);
  std::cout << "Warmup done" << std::endl;

  for (size_t i = 0; i < rounds; ++i) {
    std::cout << "Round: " << i + 1 << std::endl;
    for (size_t dim : dims) {
      auto [scalar_time_, avx2_time_, avx512f_time_, diff_avx2_,
            diff_avx512f_] = Benchmark(dim, num_iters);
      std::cout << "Dim: " << dim << " Scalar: " << scalar_time_
                << "ms AVX2: " << avx2_time_ << "ms AVX512F: " << avx512f_time_
                << "ms Diff AVX2: " << diff_avx2_
                << " Diff AVX512F: " << diff_avx512f_ << std::endl;
      // Write to CSV file
      csv_file << i + 1 << "," << dim << "," << scalar_time_ << ","
               << avx2_time_ << "," << avx512f_time_ << "," << diff_avx2_ << ","
               << diff_avx512f_ << std::endl;
    }
  }

  csv_file.close();

  return 0;
}
