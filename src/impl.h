#pragma once

#include "roxdb/db.h"
#include "vector.h"

namespace rox {

class DbImpl {
 public:
  DbImpl(const std::string &path, const Schema &schema,
         const DbOptions &options) noexcept
      : path_(path), schema_(schema), options_(options) {}
  ~DbImpl() = default;
  DbImpl(const DB &) = delete;  // non-copyable

  auto PutRecord(Key key, const Record &record) -> void;
  auto GetRecord(Key key) const -> Record;
  auto DeleteRecord(Key key) -> void;

  auto FullScan(const Query &query) const -> std::vector<QueryResult>;
  auto KnnSearch(const Query &query) const -> std::vector<QueryResult>;

 private:
  const std::string &path_;
  const Schema &schema_;
  const DbOptions &options_;
  std::unordered_map<Key, Record> records_;  // in-memory storage
  std::unordered_map<std::string, std::unique_ptr<IvfFlatIndex>> indexes_;

};  // class DbImpl

}  // namespace rox