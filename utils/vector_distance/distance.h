#pragma once

#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

// #if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
// #endif

using Float = float;
using Vector = std::vector<Float>;

namespace rox {

auto GetDistanceL2SqAvx2(const Vector &a, const Vector &b) -> Float;
auto GetDistanceL2SqAvx512F(const Vector &a, const Vector &b) -> Float;
auto GetDistanceL2SqScalar(const Vector &a, const Vector &b) -> Float;

}  // namespace rox