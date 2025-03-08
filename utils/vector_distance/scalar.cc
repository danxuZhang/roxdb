#include <numeric>

#include "distance.h"

namespace rox {

auto GetDistanceL2SqScalar(const Vector &a, const Vector &b) -> Float {
  return std::transform_reduce(
      a.begin(), a.end(), b.begin(), 0.0, std::plus<>(),
      [](Float a, Float b) { return (a - b) * (a - b); });
}

}  // namespace rox