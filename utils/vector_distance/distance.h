#pragma once

#include <cassert>
#include <vector>

using Float = float;
using Vector = std::vector<Float>;

namespace rox {

auto GetDistanceL2SqAvx2(const Vector &a, const Vector &b) -> Float;
auto GetDistanceL2SqAvx512F(const Vector &a, const Vector &b) -> Float;
auto GetDistanceL2SqScalar(const Vector &a, const Vector &b) -> Float;

}  // namespace rox