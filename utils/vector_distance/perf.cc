#include <chrono>
#include <iostream>

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

auto Benchmark(size_t dim, size_t num_iter) {
  std::vector<Float> a = GenerateRandomVector(dim);
  std::vector<Float> b = GenerateRandomVector(dim);

  for (size_t i = 0; i < num_iter; ++i) {
#if defined(__AVX2__)
    rox::GetDistanceL2SqAvx2(a, b);
#elif defined(__AVX512F__)
    rox::GetDistanceL2SqAvx512F(a, b);
#else
    rox::GetDistanceL2SqScalar(a, b);
#endif
  }
}

}  // namespace

int main() {
#if defined(__AVX2__)
  std::cout << "AVX2 is enabled" << std::endl;
#elif defined(__AVX512F__)
  std::cout << "AVX512F is enabled" << std::endl;
#else
  std::cout << "Neither AVX2 nor AVX512F is enabled" << std::endl;
#endif

  const auto dim = 1 << 12;
  const auto num_iter = 1 << 24;

  auto start = std::chrono::high_resolution_clock::now();
  Benchmark(dim, num_iter);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Dimension: " << dim << std::endl;
  std::cout << "Number of iterations: " << num_iter << std::endl;
  std::cout << "Elapsed time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;

  return 0;
}