#include <immintrin.h>

#include "distance.h"

namespace rox {

auto GetDistanceL2SqAvx512F(const Vector &a, const Vector &b) -> Float {
  constexpr const size_t kFloatsPerAvx512F = 16;
  const size_t rounds = a.size() / kFloatsPerAvx512F;
  const size_t remainder = a.size() % kFloatsPerAvx512F;

  __m512 sum = _mm512_setzero_ps();
  for (size_t i = 0; i < rounds; ++i) {
    if (i + 2 < rounds) {
      _mm_prefetch(reinterpret_cast<const char *>(
                       a.data() + ((i + 2) * kFloatsPerAvx512F)),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char *>(
                       b.data() + ((i + 2) * kFloatsPerAvx512F)),
                   _MM_HINT_T0);
    }
    const __m512 a_vec = _mm512_loadu_ps(a.data() + (i * kFloatsPerAvx512F));
    const __m512 b_vec = _mm512_loadu_ps(b.data() + (i * kFloatsPerAvx512F));
    const __m512 diff = _mm512_sub_ps(a_vec, b_vec);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  Float result = _mm512_reduce_add_ps(sum);

  if (remainder > 0) {
    const size_t start_idx = rounds * kFloatsPerAvx512F;

    // Create mask for remaining elements
    __mmask16 mask = _cvtu32_mask16((1U << remainder) - 1);

    // Masked load for boundary-safe operations
    __m512 a_rem = _mm512_maskz_loadu_ps(mask, a.data() + start_idx);
    __m512 b_rem = _mm512_maskz_loadu_ps(mask, b.data() + start_idx);
    __m512 diff_rem = _mm512_sub_ps(a_rem, b_rem);

    // Add squared differences for remainder using masked reduction
    result +=
        _mm512_mask_reduce_add_ps(mask, _mm512_mul_ps(diff_rem, diff_rem));
  }

  return result;
}

}  // namespace rox