#pragma once

#include <cstddef>

namespace rox {

struct SearchStats {
  size_t num_records_scanned = 0;
  size_t num_records_filtered_out = 0;
};

}  // namespace rox