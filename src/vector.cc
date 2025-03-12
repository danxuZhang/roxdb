#include "vector.h"

#include <algorithm>
#include <execution>
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
  std::vector<std::pair<Float, CentroidId>> distances(index_.centroids_.size());
  std::transform(std::execution::par, index_.centroids_.begin(),
                 index_.centroids_.end(), distances.begin(),
                 [&](const auto& centroid) {
                   return std::make_pair(GetDistanceL2Sq(centroid, query_),
                                         &centroid - index_.centroids_.data());
                 });

  auto comp = [](const auto& a, const auto& b) { return a.first < b.first; };
  std::partial_sort(std::execution::seq, distances.begin(),
                    distances.begin() + nprobe_, distances.end(), comp);

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

auto IvfFlatIterator::SeekCluster() -> void {
  probe_lists_.clear();
  current_prob_ = 0;

  // Calculate distance to each centroid
  std::vector<std::pair<Float, CentroidId>> distances(index_.centroids_.size());
  std::transform(std::execution::par, index_.centroids_.begin(),
                 index_.centroids_.end(), distances.begin(),
                 [&](const auto& centroid) {
                   return std::make_pair(GetDistanceL2Sq(centroid, query_),
                                         &centroid - index_.centroids_.data());
                 });

  // Sort by distance
  std::ranges::partial_sort(distances, distances.begin() + nprobe_,
                            std::less<>());

  // Add nprobe_ centroids to probe_lists_
  probe_lists_.reserve(nprobe_);
  for (size_t i = 0; i < nprobe_; ++i) {
    const auto& [_, centroid_idx] = distances[i];
    probe_lists_.push_back(centroid_idx);
  }
}

auto IvfFlatIterator::GetCluster() -> const IvfList& {
  const auto current_centroid_idx = probe_lists_[current_prob_];
  return index_.inverted_lists_[current_centroid_idx];
}

auto IvfFlatIterator::NextCluster() -> void { ++current_prob_; }

auto IvfFlatIterator::HasNextCluster() const -> bool {
  return current_prob_ < probe_lists_.size();
}

}  // namespace rox