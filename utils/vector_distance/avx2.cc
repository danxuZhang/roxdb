#include <immintrin.h>

#include "distance.h"

namespace rox {

auto GetDistanceL2SqAvx2(const Vector &a, const Vector &b) -> Float {
  constexpr const size_t kFloatsPerAvx2 = 8;
  const size_t rounds = a.size() / kFloatsPerAvx2;
  const size_t remainder = a.size() % kFloatsPerAvx2;

  __m256 sum = _mm256_setzero_ps();
  for (size_t i = 0; i < rounds; ++i) {
    const __m256 a_vec = _mm256_loadu_ps(a.data() + (i * kFloatsPerAvx2));
    const __m256 b_vec = _mm256_loadu_ps(b.data() + (i * kFloatsPerAvx2));
    const __m256 diff = _mm256_sub_ps(a_vec, b_vec);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  alignas(32) Float sum_arr[kFloatsPerAvx2];
  _mm256_store_ps(sum_arr, sum);
  Float result = 0.0;
  for (float i : sum_arr) {
    result += i;
  }

  for (size_t i = 0; i < remainder; ++i) {
    result +=
        (a[(rounds * kFloatsPerAvx2) + i] - b[(rounds * kFloatsPerAvx2) + i]) *
        (a[(rounds * kFloatsPerAvx2) + i] - b[(rounds * kFloatsPerAvx2) + i]);
  }

  return result;
}

}  // namespace rox