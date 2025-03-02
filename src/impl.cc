#include "impl.h"

#include <algorithm>
#include <compare>
#include <memory>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "roxdb/db.h"
#include "vector.h"

namespace rox {

DbImpl::DbImpl(const std::string &path, const Schema &schema,
               const DbOptions &options) noexcept
    : path_(path), schema_(schema), options_(options) {
  // Create Index, one per vector field
  for (const auto &field : schema.vector_fields) {
    indexes_[field.name] = std::make_unique<IvfFlatIndex>(field.name, field.dim,
                                                          field.num_centroids);
  }
}

auto DbImpl::PutRecord(Key key, const Record &record) -> void {
  if (records_.contains(key)) {
    throw std::invalid_argument("Record already exists");
  }
  // Add record to storage
  records_[key] = record;
  // Add record to indexes
  for (const auto &field : schema_.vector_fields) {
    const auto &vector =
        record.vectors[schema_.vector_field_idx.at(field.name)];
    indexes_.at(field.name)->Put(key, vector);
  }
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
  // Remove record from storage
  records_.erase(key);
  // Remove record from indexes
  for (const auto &field : schema_.vector_fields) {
    indexes_.at(field.name)->Delete(key);
  }
}

auto DbImpl::SetCentroids(const std::string &field,
                          const std::vector<Vector> &centroids) -> void {
  if (!indexes_.contains(field)) {
    throw std::invalid_argument("Vector field not found");
  }

  indexes_.at(field)->SetCentroids(centroids);
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

auto DbImpl::KnnSearch(const Query &query) const -> std::vector<QueryResult> {
  if (query.GetLimit() == 0) {
    return {};
  }

  // Short curcuit for single vector search
  if (query.vectors.size() == 1) {
    return SingleVectorKnnSearch(query);
  }

  return MultiVectorKnnSearch(query);
}

auto DbImpl::SingleVectorKnnSearch(const Query &query) const
    -> std::vector<QueryResult> {
  const auto k = query.GetLimit();
  const auto &[field_name, query_vec, weight] = query.GetVectors().front();
  const auto &index = *indexes_.at(field_name);

  // Create a Min Heap for top k results
  std::priority_queue<QueryResult, std::vector<QueryResult>, std::greater<>> pq;
  auto it = IvfFlatIterator(index, query_vec, options_.ivf_nprobe);

  // Iterate over the index
  for (it.Seek(); it.Valid(); it.Next()) {
    const auto key = it.GetKey();
    const auto &record_vec = it.GetVector();
    const auto distance = GetDistanceL2Sq(query_vec, record_vec);

    // Check filters
    const auto &record = records_.at(key);
    if (!std::ranges::all_of(query.GetFilters(), [&](const auto &filter) {
          return ApplyFilter(schema_, record, filter);
        })) {
      continue;
    }

    pq.push({key, distance});
    if (pq.size() == k) {
      break;
    }
  }

  std::vector<QueryResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  return results;
}

auto DbImpl::MultiVectorKnnSearch(const Query &query) const  // NOLINT
    -> std::vector<QueryResult> {
  return {};
}

}  // namespace rox