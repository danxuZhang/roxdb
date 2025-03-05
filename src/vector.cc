#include "vector.h"

#include <algorithm>
#include <utility>

#ifdef DEBUG
#include <iostream>
#endif

#include "roxdb/db.h"

namespace rox {

auto IvfFlatIterator::Seek() -> void {
  probe_lists_.clear();
  current_prob_ = 0;
  candidates_ = {};

  // Find cloest nprobe_ centroids
  std::vector<std::pair<Float, CentroidId>> distances;
  distances.reserve(index_.centroids_.size());
  for (size_t i = 0; i < index_.centroids_.size(); ++i) {
    const auto& centroid = index_.centroids_[i];
    const auto distance = GetDistanceL2Sq(centroid, query_);
    distances.emplace_back(distance, i);
  }
  auto comp = [](const auto& a, const auto& b) { return a.first < b.first; };
  std::partial_sort(distances.begin(), distances.begin() + nprobe_,
                    distances.end(), comp);

  // Add nprobe_ centroids to probe_lists_
  probe_lists_.reserve(nprobe_);
  for (size_t i = 0; i < nprobe_; ++i) {
    const auto& [_, centroid_idx] = distances[i];
    probe_lists_.push_back(centroid_idx);
  }

#ifdef DEBUG
  // Print probe clusters
  std::cout << "Probing clusters: ";
  for (const auto& idx : probe_lists_) {
    std::cout << idx << " ";
  }
  std::cout << std::endl;
#endif

  // Collect candidates from the first probe cluster
  CollectCandidates();
}

auto IvfFlatIterator::Next() -> void {
  candidates_.pop();
  while (candidates_.empty()) {
    ++current_prob_;
    if (current_prob_ >= nprobe_) {
      return;
    }
    CollectCandidates();
  }
}

auto IvfFlatIterator::CollectCandidates() -> void {
  candidates_ = {};
  const auto current_centroid_idx = probe_lists_[current_prob_];

#ifdef DEBUG
  std::cout << "Collecting candidates from cluster " << current_centroid_idx
            << std::endl;
#endif
  for (const auto& [key, vector] :
       index_.inverted_lists_[current_centroid_idx]) {
    const auto distance = GetDistanceL2Sq(vector, query_);
    candidates_.push({key, vector, distance});
  }
}

auto IvfFlatIterator::Valid() const -> bool {
  return current_prob_ < probe_lists_.size() && !candidates_.empty();
}

auto IvfFlatIterator::GetKey() const noexcept -> Key {
  return candidates_.top().key;
}

auto IvfFlatIterator::GetVector() const noexcept -> const Vector& {
  return candidates_.top().vector;
}

}  // namespace rox