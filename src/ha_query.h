#pragma once

#include <mutex>

#include "impl.h"
#include "roxdb/db.h"

namespace rox {

class QueryHandler {
 public:
  QueryHandler(const DbImpl &db, const Query &query) : db_(db), query_(query) {}

  auto KnnSearch(size_t nprobe) -> std::vector<QueryResult>;

  auto KnnSearchIterativeMerge(size_t nprobe, size_t k_threshold) -> std::vector<QueryResult>;

 private:
  struct Iterator {
    const std::string &field;
    const Vector &query;
    const Float weight;
    std::unique_ptr<IvfFlatIterator> it;
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();
    Float last_seen_distance = std::numeric_limits<Float>::max();

    // Constructor
    Iterator(const std::string &field, const Vector &query, Float weight,
             std::unique_ptr<IvfFlatIterator> it)
        : field(field), query(query), weight(weight), it(std::move(it)) {}
  };

  const DbImpl &db_;
  const Query &query_;

  auto GetTopK(const std::string &field, const Vector &query, size_t k,
               size_t nprobe) const -> std::vector<Key>;

};  // class QueryHandler

}  // namespace rox