#include "impl.h"

#include <algorithm>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "roxdb/db.h"

namespace rox {

auto DbImpl::PutRecord(Key key, const Record &record) -> void {
  if (records_.contains(key)) {
    throw std::invalid_argument("Record already exists");
  }

  records_[key] = record;
}

auto DbImpl::GetRecord(Key key) const -> Record {
  if (!records_.contains(key)) {
    throw std::invalid_argument("Record not found");
  }

  return records_.at(key);
}

auto DbImpl::DeleteRecord(Key key) -> void {
  if (!records_.contains(key)) {
    throw std::invalid_argument("Record not found");
  }

  records_.erase(key);
}

auto DbImpl::FullScan(const Query &query) const -> std::vector<QueryResult> {
  if (query.GetLimit() == 0) {
    return {};
  }

  auto records =
      records_ |  // Filter records based on scalar filters
      std::views::filter([&](const auto &pair) {
        const auto &[key, record] = pair;
        return std::ranges::all_of(query.filters, [&](const auto &filter) {
          return ApplyFilter(schema_, record, filter);
        });
      }) |  // Calculate distance for each record
      std::views::transform([&](const auto &pair) {
        const auto &[key, record] = pair;
        // Calculate aggregate distance
        Float distance = 0.0;
        for (const auto &[field, query_vector, weight] : query.vectors) {
          const auto &record_vector =
              record.vectors[schema_.vector_field_idx.at(field)];
          distance += GetDistanceL2Sq(query_vector, record_vector) * weight;
        }

        return QueryResult{key, distance};
      });

  // Find top k records
  // Create a max heap pq, the top element is the largest
  // top() is the largest, pop() removes the largest in the heap
  // New candidate only needs to compare with the largest in the heap (top)
  std::priority_queue<QueryResult> pq;
  for (const auto &result : records) {
    // pq.emplace(result);
    if (pq.size() < query.limit) {
      pq.push(result);
    } else if (result.distance < pq.top().distance) {
      pq.pop();
      pq.push(result);
    }
  }

  std::vector<QueryResult> results;
  results.reserve(query.GetLimit());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  // Reverse the results to get the smallest distance first
  std::ranges::reverse(results);
  return results;
}

auto DbImpl::KnnSearch(const Query &query) const  // NOLINT
    -> std::vector<QueryResult> {
  return {};
}

}  // namespace rox