#pragma once

#include <cassert>
#include <numeric>

#include "roxdb/db.h"

namespace rox {

inline auto GetDistanceL2Sq(const Vector &a, const Vector &b) noexcept
    -> Float {
  return std::transform_reduce(
      a.begin(), a.end(), b.begin(), 0.0, std::plus<>(),
      [](Float x, Float y) { return (x - y) * (x - y); });
}

inline auto GetDistanceL1(const Vector &a, const Vector &b) noexcept -> Float {
  return std::transform_reduce(
      a.begin(), a.end(), b.begin(), 0.0, std::plus<>(),
      [](Float x, Float y) { return std::abs(x - y); });
}

}  // namespace rox