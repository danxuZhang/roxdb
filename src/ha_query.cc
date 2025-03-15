#include "ha_query.h"

#include "impl.h"
#include "roxdb/db.h"

namespace rox {

auto QueryHandler::KnnSearch(size_t nprobe) -> std::vector<QueryResult> {
  const auto k = query_.GetLimit();
  const auto &query_vectors = query_.GetVectors();

  // Create Iterators for each vector field
  std::vector<Iterator> its;
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

}  // namespace rox