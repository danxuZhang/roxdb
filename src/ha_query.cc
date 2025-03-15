#include "ha_query.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "impl.h"
#include "roxdb/db.h"
#include "vector.h"

namespace rox {

auto QueryHandler::KnnSearch(size_t nprobe) -> std::vector<QueryResult> {
  const auto k = query_.GetLimit();
  const auto &query_vectors = query_.GetVectors();

  // Create Iterators for each vector field
  std::vector<AptIterator> its;
  for (const auto &[field_name, query_vec, weight] : query_vectors) {
    const auto &index = *db_.indexes_.at(field_name);
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

            const auto &record = db_.storage_->GetRecord(key);

            // Check filters
            if (query_.GetFilters().size() > 0) {
              if (!std::ranges::all_of(
                      query_.GetFilters(), [&](const auto &filter) {
                        return ApplyFilter(db_.schema_, record, filter);
                      })) {
                return;
              }
            }

            // Calculate total distance
            Float total_distance = 0.0;
            for (const auto &[field_name, query_vec, weight] : query_vectors) {
              const auto &record_vec =
                  record.vectors[db_.schema_.vector_field_idx.at(field_name)];
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
  // std::ranges::reverse(results);
  return results;
}

auto QueryHandler::GetTopK(const std::string &field, const Vector &query,
                           size_t k, size_t nprobe) const -> std::vector<Key> {
  const auto &idx = db_.indexes_.at(field);
  auto it = std::make_unique<IvfFlatIterator>(*idx, query, nprobe, 0, 0);

  std::priority_queue<QueryResult> pq;
  it->Seek();
  while (it->Valid()) {
    const auto key = it->GetKey();
    const auto &record_vec = it->GetVector();
    const auto distance = GetDistanceL2Sq(query, record_vec);
    it->Next();

    if (pq.size() < k) {
      pq.push({key, distance});
    } else if (distance < pq.top().distance) {
      pq.pop();
      pq.push({key, distance});
    }
  }

  std::vector<Key> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back(pq.top().id);
    pq.pop();
  }
  return results;
}

auto QueryHandler::KnnSearchIterativeMerge(size_t nprobe, size_t k_threshold)
    -> std::vector<QueryResult> {
  auto ori_k = query_.GetLimit();
  auto k = query_.GetLimit();
  const auto &query_vectors = query_.GetVectors();

  std::priority_queue<QueryResult> pq;  // global result heap
  std::unordered_set<Key> visited;      // Avoid duplicate keys

  // Threshold Algorithm Setup
  std::unordered_map<std::string, Float>
      threshold_values;  // field -> threshold
  for (const auto &[field_name, query_vec, weight] : query_vectors) {
    threshold_values[field_name] = std::numeric_limits<Float>::max();
  }

  while (k < k_threshold) {
    // run top-k search for each field
    std::unordered_set<Key> candidates;
    for (const auto &[field_name, query_vec, weight] : query_vectors) {
      const auto topk = GetTopK(field_name, query_vec, k, nprobe);
      for (const auto &key : topk) {
        if (visited.contains(key)) {
          continue;
        }
        candidates.insert(key);
      }
    }  // for (const auto &[field_name, query_vec, weight] : query_vectors)

    // calculate total distance for each candidate, apply filter, update
    // threshold
    for (const auto &key : candidates) {
      const auto &record = db_.storage_->GetRecord(key);
      if (query_.GetFilters().size() > 0) {
        if (!std::ranges::all_of(query_.GetFilters(), [&](const auto &filter) {
              return ApplyFilter(db_.schema_, record, filter);
            })) {
          continue;
        }
      }

      Float total_distance = 0.0;
      for (const auto &[field_name, query_vec, weight] : query_vectors) {
        const auto &record_vec =
            record.vectors[db_.schema_.vector_field_idx.at(field_name)];
        total_distance += GetDistanceL2Sq(query_vec, record_vec) * weight;
      }
      visited.insert(key);

      // Update threshold values
      for (const auto &[field_name, query_vec, weight] : query_vectors) {
        const auto &record_vec =
            record.vectors[db_.schema_.vector_field_idx.at(field_name)];
        const auto distance = GetDistanceL2Sq(query_vec, record_vec);
        threshold_values[field_name] =
            std::min(threshold_values[field_name], distance);
      }

      // Try to insert into the heap
      if (pq.size() < ori_k) {
        pq.push({key, total_distance});
      } else if (total_distance < pq.top().distance) {
        pq.pop();
        pq.push({key, total_distance});
      }  // for (const auto &key : candidates)
    }
    // Check Threshold Algorithm stopping condition
    Float distance_sum = 0.0;
    for (const auto &[field_name, query_vec, weight] : query_vectors) {
      distance_sum += threshold_values[field_name] * weight;
    }
    if (pq.size() == ori_k && distance_sum >= pq.top().distance) {
      break;
    }

    // if not met, double k and continue
    k *= 2;
  }  // while (k < k_threshold)

  std::vector<QueryResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  return results;
}

auto QueryHandler::KnnSearchVBase(size_t nprobe, size_t n2)
    -> std::vector<QueryResult> {
  constexpr const size_t kKPerRound = 1;
  const auto k = query_.GetLimit();
  const auto &query_vectors = query_.GetVectors();

  // Iterator for each field
  std::vector<IvfFlatIterator> its;
  for (const auto &[field_name, query_vec, weight] : query_vectors) {
    const auto &index = *db_.indexes_.at(field_name);
    auto it = IvfFlatIterator(index, query_vec, nprobe, 0, 0);
    it.Seek();
    its.push_back(std::move(it));
  }

  // Average score tracking for each field
  std::unordered_map<std::string, Float> scores_sum;
  std::unordered_map<std::string, size_t> scores_count;
  for (const auto &[field_name, query_vec, weight] : query_vectors) {
    scores_sum[field_name] = 0.0F;
    scores_count[field_name] = 0;
  }

  // Threshold Algorithm Setup
  std::unordered_map<std::string, Float>
      threshold_values;  // field -> threshold
  for (const auto &[field_name, query_vec, weight] : query_vectors) {
    threshold_values[field_name] = std::numeric_limits<Float>::max();
  }

  std::priority_queue<QueryResult> pq;
  std::unordered_set<Key> visited;
  while (true) {
    // Calculate steps for each field
    std::unordered_map<std::string, size_t> steps;
    if (n2 == 0 ||
        std::ranges::min_element(scores_count, {},
                                 &std::pair<const std::string, size_t>::second)
                ->second == 0) {
      for (const auto &[field_name, query_vec, weight] : query_vectors) {
        steps[field_name] = kKPerRound;
      }
    } else {
      float avg_score_reciprocal = 0.0F;
      for (const auto &[field_name, query_vec, weight] : query_vectors) {
        avg_score_reciprocal +=
            1.0F / (scores_sum[field_name] / scores_count[field_name]);
      }
      for (const auto &[field_name, query_vec, weight] : query_vectors) {
        steps[field_name] =
            kKPerRound +
            std::ceil(n2 * (scores_count[field_name] / scores_sum[field_name]) /
                      avg_score_reciprocal);
      }
    }

    // Run top-k search for each field
    bool exhausted = true;
    for (size_t field_i = 0; field_i < query_vectors.size(); ++field_i) {
      const auto &[field_name, query_vec, weight] = query_vectors[field_i];
      auto &it = its[field_i];
      if (!it.Valid()) {
        continue;
      }
      exhausted = false;

      // Calculate the number of steps for this field
      const auto &step = steps[field_name];
      for (size_t i = 0; i < step; ++i) {
        if (!it.Valid()) {
          break;
        }

        const auto key = it.GetKey();
        const auto &record_vec = it.GetVector();
        const auto distance = GetDistanceL2Sq(query_vec, record_vec);
        it.Next();

        {  // Skip if key is already visited
          if (!visited.insert(key).second) {
            continue;
          }
        }

        const auto &record = db_.storage_->GetRecord(key);
        // Apply filters
        if (query_.GetFilters().size() > 0) {
          if (!std::ranges::all_of(
                  query_.GetFilters(), [&](const auto &filter) {
                    return ApplyFilter(db_.schema_, record, filter);
                  })) {
            continue;
          }
        }

        // Calculate total distance
        Float total_distance = 0.0;
        for (const auto &[field_name, query_vec, weight] : query_vectors) {
          const auto &record_vec =
              record.vectors[db_.schema_.vector_field_idx.at(field_name)];
          total_distance += GetDistanceL2Sq(query_vec, record_vec) * weight;
        }

        // Update last seen distance
        threshold_values[field_name] =
            std::min(threshold_values[field_name], distance);

        // Update scores
        scores_sum[field_name] += total_distance;
        scores_count[field_name] += 1;

        // Try to insert into the heap
        if (pq.size() < k) {
          pq.push({key, total_distance});
        } else if (total_distance < pq.top().distance) {
          pq.pop();
          pq.push({key, total_distance});
        }
      }  // for step times
    }  // for field in query_vectors
    if (exhausted) {
      break;
    }

    // Check Threshold Algorithm stopping condition
    Float distance_sum = 0.0;
    for (const auto &[field_name, query_vec, weight] : query_vectors) {
      distance_sum += threshold_values[field_name] * weight;
    }
    if (pq.size() == k && distance_sum >= pq.top().distance) {
      break;
    }
  }  // while (true)

  std::vector<QueryResult> results;
  results.reserve(pq.size());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  return results;
}

}  // namespace rox