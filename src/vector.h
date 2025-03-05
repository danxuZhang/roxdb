#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include "roxdb/db.h"

namespace rox {

inline auto GetDistanceL2Sq(const Vector &a, const Vector &b) noexcept
    -> Float {
  return std::transform_reduce(
      a.begin(), a.end(), b.begin(), 0.0, std::plus<>(),
      [](Float x, Float y) { return (x - y) * (x - y); });
}

inline auto GetDistanceL1(const Vector &a, const Vector &b) noexcept -> Float {
  return std::transform_reduce(
      a.begin(), a.end(), b.begin(), 0.0, std::plus<>(),
      [](Float x, Float y) { return std::abs(x - y); });
}

using CentroidId = size_t;
using IvfList = std::vector<std::pair<Key, Vector>>;

inline auto AssignCentroid(const Vector &v,
                           const std::vector<Vector> &centroids,
                           const size_t dim) noexcept -> CentroidId {
  assert(!centroids.empty());
  assert(v.size() == dim);
  CentroidId best_cluster = 0;
  Float best_distance = GetDistanceL2Sq(v, centroids[0]);
  for (CentroidId cluster = 1; cluster < centroids.size(); ++cluster) {
    const Float distance = GetDistanceL2Sq(v, centroids[cluster]);
    if (distance < best_distance) {
      best_cluster = cluster;
      best_distance = distance;
    }
  }
  return best_cluster;
}

class IvfFlatIndex {
 public:
  IvfFlatIndex(const std::string &field_name, const size_t dim,
               const size_t nlist)
      : field_name_(field_name), dim_(dim), nlist_(nlist) {
    centroids_.resize(nlist_);
    inverted_lists_.resize(nlist_);
  }

  auto Put(const Key &key, const Vector &v) -> void {
    const CentroidId cluster = AssignCentroid(v, centroids_, dim_);
    inverted_lists_[cluster].emplace_back(key, v);
  }

  auto Delete(const Key &key) -> void {
    for (auto &list : inverted_lists_) {
      auto it = std::ranges::remove_if(list, [&key](const auto &pair) {
                  return pair.first == key;
                }).begin();
      list.erase(it, list.end());
    }
  }

  auto SetCentroids(const std::vector<Vector> &centroids) -> void {
    assert(centroids.size() == nlist_);
    centroids_ = centroids;
  }

  auto SetInvertedLists(const std::vector<IvfList> &inverted_lists) -> void {
    assert(inverted_lists.size() == nlist_);
    inverted_lists_ = inverted_lists;
  }

  auto GetCentroids() const noexcept -> const std::vector<Vector> & {
    return centroids_;
  }

  auto GetInvertedLists() const noexcept -> const std::vector<IvfList> & {
    return inverted_lists_;
  }

  auto GetName() const noexcept -> const std::string & { return field_name_; }

 private:
  friend class IvfFlatIterator;
  const std::string &field_name_;
  const size_t dim_;
  const size_t nlist_;

  std::vector<Vector> centroids_;
  std::vector<IvfList> inverted_lists_;
};  // class IvfFlatIndex

class IvfFlatIterator {
 public:
  IvfFlatIterator(const IvfFlatIndex &index, const Vector &query,
                  const size_t nprobe)
      : index_(index), query_(query), nprobe_((nprobe)) {}

  auto Seek() -> void;

  auto Next() -> void;

  auto Valid() const -> bool;

  auto GetKey() const noexcept -> Key;
  auto GetVector() const noexcept -> const Vector &;

 private:
  struct Candidate {
    Key key;
    Vector vector;
    Float distance;

    auto operator<=>(const Candidate &other) const {
      return distance <=> other.distance;
    }
  };
  const IvfFlatIndex &index_;
  const Vector &query_;
  const size_t nprobe_;

  std::vector<CentroidId> probe_lists_;  // clusters to probe
  size_t current_prob_;                  // current probe cluster index
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>>
      candidates_;  // candidates in the current probe cluster (min heap)

  auto CollectCandidates() -> void;
};  // class IvfFlatIterator

}  // namespace rox