#include "impl.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "roxdb/db.h"
#include "storage.h"
#include "vector.h"

namespace rox {

DbImpl::DbImpl(const std::string &path, const DbOptions &options)
    : path_(path), options_(options) {
  if (options.create_if_missing) {
    throw std::invalid_argument(
        "Can only open existing database without Schema");
  }
  storage_ = std::make_unique<Storage>(path, options);

  // Load schema
  schema_ = storage_->GetSchema();
  // DEBUG: Print schema fields
  // for (const auto &field : schema_.vector_fields) {
  //   std::cout << "Vector Field: " << field.name << " " << field.dim << " "
  //             << field.num_centroids << std::endl;
  // }

  // Load indexes
  for (const auto &field : schema_.vector_fields) {
    indexes_[field.name] = storage_->GetIndex(field.name);
  }
  // Populate schema idx maps
  for (size_t i = 0; i < schema_.vector_fields.size(); ++i) {
    schema_.vector_field_idx[schema_.vector_fields[i].name] = i;
  }
  for (size_t i = 0; i < schema_.scalar_fields.size(); ++i) {
    schema_.scalar_field_idx[schema_.scalar_fields[i].name] = i;
  }
  // Preload records
  storage_->PrefetchRecords(1000);
}

DbImpl::DbImpl(const std::string &path, const DbOptions &options,
               const Schema &schema) noexcept
    : path_(path), options_(options), schema_(schema) {
  // Create Index, one per vector field
  for (const auto &field : schema.vector_fields) {
    indexes_[field.name] = std::make_unique<IvfFlatIndex>(field.name, field.dim,
                                                          field.num_centroids);
  }
  // Create Storage
  storage_ = std::make_unique<Storage>(path, options);
  storage_->PutSchema(schema_);
}

DbImpl::~DbImpl() {
  // Save indexes
  for (const auto &[field, index] : indexes_) {
    storage_->PutIndex(field, *index);
  }
  // Save records
  storage_->FlushRecords();

  std::cout << "Cache hit: " << storage_->GetCacheHit() << std::endl;
  std::cout << "Cache miss: " << storage_->GetCacheMiss() << std::endl;
}

auto DbImpl::PutRecord(Key key, const Record &record) -> void {
  // Add record to storage
  storage_->PutRecord(key, record);
  // Add record to indexes
  for (const auto &field : schema_.vector_fields) {
    const auto &vector =
        record.vectors[schema_.vector_field_idx.at(field.name)];
    indexes_.at(field.name)->Put(key, vector);
  }
}

auto DbImpl::GetRecord(Key key) const -> Record {
  return storage_->GetRecord(key);
}

auto DbImpl::DeleteRecord(Key key) -> void {
  // Remove record from storage
  storage_->DeleteRecord(key);
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

auto DbImpl::FlushRecords() -> void { storage_->FlushRecords(); }

auto DbImpl::FullScan(const Query &query) const -> std::vector<QueryResult> {
  if (query.GetLimit() == 0) {
    return {};
  }

  // auto records =
  //     records_ |  // Filter records based on scalar filters
  //     std::views::filter([&](const auto &pair) {
  //       const auto &[key, record] = pair;
  //       return std::ranges::all_of(query.filters, [&](const auto &filter) {
  //         return ApplyFilter(schema_, record, filter);
  //       });
  //     }) |  // Calculate distance for each record
  //     std::views::transform([&](const auto &pair) {
  //       const auto &[key, record] = pair;
  //       // Calculate aggregate distance
  //       Float distance = 0.0;
  //       for (const auto &[field, query_vector, weight] : query.vectors) {
  //         const auto &record_vector =
  //             record.vectors[schema_.vector_field_idx.at(field)];
  //         distance += GetDistanceL2Sq(query_vector, record_vector) * weight;
  //       }

  //       return QueryResult{key, distance};
  //     });

  // Find top k records
  // Create a max heap pq, the top element is the largest
  // top() is the largest, pop() removes the largest in the heap
  // New candidate only needs to compare with the largest in the heap (top)
  std::priority_queue<QueryResult> pq;

  for (auto it = storage_->GetIterator(RdbStorage::kRecordPrefix); it->Valid();
       it->Next()) {
    const auto rdb_key = it->key();
    std::string_view key_view(rdb_key.data(), rdb_key.size());
    if (!key_view.starts_with(RdbStorage::kRecordPrefix)) {
      break;  // Skip keys that don't have the correct prefix
    }
    const auto key = RdbStorage::GetKey(rdb_key);
    const auto record = storage_->GetRecord(key);

    // Filter records based on scalar filters
    if (!std::ranges::all_of(query.GetFilters(), [&](const auto &filter) {
          return ApplyFilter(schema_, record, filter);
        })) {
      continue;
    }

    // Calculate distance for each record
    Float distance = 0.0;
    for (const auto &[field_name, query_vec, weight] : query.GetVectors()) {
      const auto &record_vec =
          record.vectors[schema_.vector_field_idx.at(field_name)];
      assert(query_vec.size() == record_vec.size());
      distance += GetDistanceL2Sq(query_vec, record_vec) * weight;
    }

    QueryResult result{.id = key, .distance = distance};

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

auto DbImpl::KnnSearch(const Query &query, size_t nprobe) const
    -> std::vector<QueryResult> {
  if (query.GetLimit() == 0) {
    return {};
  }

  // // Short curcuit for single vector search
  // if (query.vectors.size() == 1) {
  //   return SingleVectorKnnSearch(query, nprobe);
  // }

  return MultiVectorKnnSearch(query, nprobe);
}

auto DbImpl::SingleVectorKnnSearch(const Query &query, size_t nprobe) const
    -> std::vector<QueryResult> {
  const auto k = query.GetLimit();
  const auto &[field_name, query_vec, weight] = query.GetVectors().front();
  const auto &index = *indexes_.at(field_name);

  // Create a Max Heap for top k results
  std::priority_queue<QueryResult> pq;
  auto it = IvfFlatIterator(index, query_vec, nprobe, 0, 0);

  // Iterate over the index
  for (it.Seek(); it.Valid(); it.Next()) {
    const auto key = it.GetKey();
    const auto &record_vec = it.GetVector();
    const auto distance = GetDistanceL2Sq(query_vec, record_vec);

    // Check filters
    if (query.GetFilters().size() > 0) {
      const auto &record = storage_->GetRecord(key);
      if (!std::ranges::all_of(query.GetFilters(), [&](const auto &filter) {
            return ApplyFilter(schema_, record, filter);
          })) {
        continue;
      }
    }

    // Try to insert into the heap
    if (pq.size() < k) {
      pq.push({key, distance});
    } else if (distance < pq.top().distance) {
      pq.pop();
      pq.push({key, distance});
    }
  }

  std::vector<QueryResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  // Reverse the results to get the smallest distance first
  std::ranges::reverse(results);
  return results;
}

auto DbImpl::MultiVectorKnnSearch(const Query &query, size_t nprobe) const
    -> std::vector<QueryResult> {
  const auto k = query.GetLimit();
  const auto &query_vectors = query.GetVectors();

  // Create Iterators for each vector field
  std::vector<Iter> its;
  for (const auto &[field_name, query_vec, weight] : query_vectors) {
    const auto &index = *indexes_.at(field_name);
    auto it = std::make_unique<IvfFlatIterator>(index, query_vec, nprobe, 0, 0);
    it->SeekCluster();
    its.emplace_back(field_name, query_vec, weight, std::move(it));
  }

  // Create a Max Heap for top k results
  // The top element is the largest
  // top() is the largest, pop() removes the largest in the heap
  // New candidate only needs to compare with the largest in the heap (top)
  std::priority_queue<QueryResult> pq;
  std::mutex pq_mutex;
  std::unordered_set<Key> visited;  // Avoid duplicate keys
  std::mutex visited_mutex;

  while (true) {
    bool exhausted = true;
    // Iterate over the iterators one round, one cluster at a time
    for (auto &it : its) {
      if (!it.it->HasNextCluster()) {
        continue;
      }
      exhausted = false;

      const auto &cluster = it.it->GetCluster();
      std::for_each(
          std::execution::par, cluster.begin(), cluster.end(),
          [&](const auto &pair) {
            const auto key = pair.first;
            const auto &record_vec = pair.second;
            const auto &distance = GetDistanceL2Sq(it.query, record_vec);

            {  // Skip if key is already visited
              std::lock_guard<std::mutex> lock(visited_mutex);
              if (!visited.insert(key).second) {
                return;
              }
            }

            const auto &record = storage_->GetRecord(key);

            // Check filters
            if (query.GetFilters().size() > 0) {
              if (!std::ranges::all_of(
                      query.GetFilters(), [&](const auto &filter) {
                        return ApplyFilter(schema_, record, filter);
                      })) {
                return;
              }
            }

            // Calculate total distance
            Float total_distance = 0.0;
            for (const auto &[field_name, query_vec, weight] : query_vectors) {
              const auto &record_vec =
                  record.vectors[schema_.vector_field_idx.at(field_name)];
              total_distance += GetDistanceL2Sq(query_vec, record_vec) * weight;
            }

            // Update last seen distance
            {
              std::lock_guard<std::mutex> lock(*it.mutex);
              it.last_seen_distance = std::min(it.last_seen_distance, distance);
            }

            // Try to insert into the heap
            std::lock_guard<std::mutex> lock(pq_mutex);
            if (pq.size() < k) {
              pq.push({key, total_distance});
            } else if (total_distance < pq.top().distance) {
              pq.pop();
              pq.push({key, total_distance});
            }
          });

      it.it->NextCluster();
    }  // for (auto &it : its)

    // Check Threshold Algorithm stopping condition
    Float distance_sum = 0.0;
    for (const auto &it : its) {
      distance_sum += it.last_seen_distance * it.weight;
    }
    if (pq.size() == k && distance_sum >= pq.top().distance) {
      break;
    }

    if (exhausted) {
      break;
    }
  }  // while (true)

  std::vector<QueryResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  // Reverse the results to get the smallest distance first
  std::ranges::reverse(results);
  return results;
}

}  // namespace rox