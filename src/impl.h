#pragma once

#include <limits>
#include <memory>

#include "roxdb/db.h"
#include "vector.h"

namespace rox {

class DbImpl {
 public:
  DbImpl(const std::string &path, const Schema &schema,
         const DbOptions &options) noexcept;
  ~DbImpl() = default;
  DbImpl(const DB &) = delete;  // non-copyable

  auto PutRecord(Key key, const Record &record) -> void;
  auto GetRecord(Key key) const -> Record;
  auto DeleteRecord(Key key) -> void;

  auto SetCentroids(const std::string &field,
                    const std::vector<Vector> &centroids) -> void;

  auto FullScan(const Query &query) const -> std::vector<QueryResult>;
  auto KnnSearch(const Query &query) const -> std::vector<QueryResult>;

 private:
  const std::string &path_;
  const Schema &schema_;
  const DbOptions &options_;
  std::unordered_map<Key, Record> records_;  // in-memory storage
  std::unordered_map<std::string, std::unique_ptr<IvfFlatIndex>> indexes_;

  auto SingleVectorKnnSearch(const Query &query) const
      -> std::vector<QueryResult>;

  struct Iter {
    const std::string &field;
    const Vector &query;
    const Float weight;
    std::unique_ptr<IvfFlatIterator> it;
    Float last_seen_distance = std::numeric_limits<Float>::max();
  };  // Iterator for Faign's Threshold Algorithm
  auto MultiVectorKnnSearch(const Query &query) const
      -> std::vector<QueryResult>;
};  // class DbImpl

}  // namespace rox