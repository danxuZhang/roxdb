#pragma once

#include <string_view>

namespace rox {

class DB {
  constexpr static const char *kVersion = "0.1.0";

 public:
  static auto GetVersion() -> std::string_view { return kVersion; }

  DB() = default;
  ~DB() = default;
  DB(const DB &) = delete;  // non-copyable

};  // class DB

}  // namespace rox