#include <immintrin.h>

#include "distance.h"

namespace rox {

auto GetDistanceL2SqAvx512F(const Vector &a, const Vector &b) -> Float {
  constexpr const size_t kFloatsPerAvx512F = 16;
  const size_t rounds = a.size() / kFloatsPerAvx512F;
  const size_t remainder = a.size() % kFloatsPerAvx512F;

  __m512 sum = _mm512_setzero_ps();
  for (size_t i = 0; i < rounds; ++i) {
    const __m512 a_vec = _mm512_loadu_ps(a.data() + (i * kFloatsPerAvx512F));
    const __m512 b_vec = _mm512_loadu_ps(b.data() + (i * kFloatsPerAvx512F));
    const __m512 diff = _mm512_sub_ps(a_vec, b_vec);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  alignas(64) Float sum_arr[kFloatsPerAvx512F];
  _mm512_store_ps(sum_arr, sum);
  Float result = 0.0;
  for (float i : sum_arr) {
    result += i;
  }

  for (size_t i = 0; i < remainder; ++i) {
    result += (a[(rounds * kFloatsPerAvx512F) + i] -
               b[(rounds * kFloatsPerAvx512F) + i]) *
              (a[(rounds * kFloatsPerAvx512F) + i] -
               b[(rounds * kFloatsPerAvx512F) + i]);
  }

  return result;
}

}  // namespace rox