#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "roxdb/db.h"
#include "vector_distance.h"

namespace rox {

using CentroidId = size_t;
using IvfList = std::vector<std::pair<Key, Vector>>;

inline auto AssignCentroid(const Vector &v,
                           const std::vector<Vector> &centroids,
                           const size_t dim) noexcept -> CentroidId {
  assert(!centroids.empty());
  assert(v.size() == dim);
  std::vector<Float> distances(centroids.size());

  std::transform(std::execution::par, centroids.begin(), centroids.end(),
                 distances.begin(), [&v](const auto &centroid) {
                   return GetDistanceL2Sq(centroid, v);
                 });

  return std::distance(distances.begin(), std::ranges::min_element(distances));
}

class IvfFlatIndex {
 public:
  IvfFlatIndex(std::string field_name, const size_t dim, const size_t nlist)
      : field_name_(std::move(field_name)), dim_(dim), nlist_(nlist) {
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
  friend class RdbStorage;
  const std::string field_name_;
  const size_t dim_;
  const size_t nlist_;

  std::vector<Vector> centroids_;
  std::vector<IvfList> inverted_lists_;
};  // class IvfFlatIndex

class IvfFlatIterator {
 public:
  IvfFlatIterator(const IvfFlatIndex &index, const Vector &query, size_t nprobe,
                  size_t rm_window_size [[maybe_unused]],
                  size_t rm_neighbor_size [[maybe_unused]])
      : index_(index), query_(query), nprobe_((nprobe)) {}

  auto Seek() -> void;

  auto Next() -> void;

  auto Valid() const -> bool;

  auto GetKey() const noexcept -> Key;
  auto GetVector() const noexcept -> const Vector &;

  auto SeekCluster() -> void;
  auto NextCluster() -> void;
  auto GetCluster() -> const IvfList &;
  auto HasNextCluster() const -> bool;

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
  size_t current_prob_ = 0;              // current probe cluster index
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>>
      candidates_;  // candidates in the current probe cluster (min heap)

  auto CollectCandidates() -> void;
};  // class IvfFlatIterator

}  // namespace rox