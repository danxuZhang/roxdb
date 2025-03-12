#pragma once

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "roxdb/db.h"
#include "storage.h"
#include "vector.h"

namespace rox {

class DbImpl {
 public:
  explicit DbImpl(const std::string &path, const DbOptions &options);
  explicit DbImpl(const std::string &path, const DbOptions &options,
                  const Schema &schema) noexcept;
  ~DbImpl();
  DbImpl(const DB &) = delete;             // non-copyable
  DbImpl &operator=(const DB &) = delete;  // non-assignable

  auto PutRecord(Key key, const Record &record) -> void;
  auto GetRecord(Key key) const -> Record;
  auto DeleteRecord(Key key) -> void;
  auto FlushRecords() -> void;

  auto SetCentroids(const std::string &field,
                    const std::vector<Vector> &centroids) -> void;

  auto FullScan(const Query &query) const -> std::vector<QueryResult>;
  auto KnnSearch(const Query &query, size_t nprobe) const
      -> std::vector<QueryResult>;

 private:
  const std::string &path_;
  const DbOptions &options_;
  Schema schema_;
  // std::unordered_map<Key, Record> records_;  // in-memory storage
  std::unique_ptr<Storage> storage_;
  std::unordered_map<std::string, std::unique_ptr<IvfFlatIndex>> indexes_;
  std::unordered_map<std::string, size_t> index_avg_size_;

  auto SingleVectorKnnSearch(const Query &query, size_t nprobe) const
      -> std::vector<QueryResult>;

  struct Iter {
    const std::string &field;
    const Vector &query;
    const Float weight;
    std::unique_ptr<IvfFlatIterator> it;
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();
    Float last_seen_distance = std::numeric_limits<Float>::max();

    // Constructor
    Iter(const std::string &field, const Vector &query, Float weight,
         std::unique_ptr<IvfFlatIterator> it)
        : field(field), query(query), weight(weight), it(std::move(it)) {}
  };  // Iterator for Faign's Threshold Algorithm
  auto MultiVectorKnnSearch(const Query &query, size_t nprobe) const
      -> std::vector<QueryResult>;
};  // class DbImpl

}  // namespace rox