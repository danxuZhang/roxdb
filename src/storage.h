#pragma once

#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include <memory>
#include <unordered_map>

#include "rocksdb/slice.h"
#include "roxdb/db.h"
#include "vector.h"

namespace rox {

class RdbStorage;

class Storage {
 public:
  explicit Storage(std::string_view path, const DbOptions& options);

  // Pass-through to RdbStorage
  auto PutSchema(const Schema& schema) -> void;
  // Pass-through to RdbStorage
  auto GetSchema() const -> Schema;

  // Cached in-memory
  auto PutRecord(Key key, const Record& record) -> void;
  // Cached in-memory
  auto GetRecord(Key key) -> Record;
  // Write-through
  auto DeleteRecord(Key key) -> void;

  // Cache records in memory
  auto PrefetchRecords(size_t n) -> void;
  // Flush cached records to RdbStorage
  auto FlushRecords() -> void;

  // Pass-through to RdbStorage
  auto PutIndex(const std::string& field, const IvfFlatIndex& index) -> void;
  // Pass-through to RdbStorage
  auto GetIndex(const std::string& field) -> std::unique_ptr<IvfFlatIndex>;
  // Pass-through to RdbStorage
  auto DeleteIndex(const std::string& field) -> void;

  // Pass-through to RdbStorage
  auto GetIterator(std::string_view prefix)
      -> std::unique_ptr<rocksdb::Iterator>;

  auto GetCacheHit() const noexcept -> size_t { return cache_hit_; }
  auto GetCacheMiss() const noexcept -> size_t { return cache_miss_; }

 private:
  size_t cache_hit_ = 0;
  size_t cache_miss_ = 0;
  std::unordered_map<Key, Record> records_cache_;
  std::unique_ptr<RdbStorage> rdb_storage_;
};

class RdbStorage {
 public:
  explicit RdbStorage(std::string_view path, const DbOptions& options);

  ~RdbStorage();

  auto PutSchema(const Schema& schema) -> void;
  auto GetSchema() const -> Schema;

  auto PutRecord(Key key, const Record& record) -> void;
  auto GetRecord(Key key) const -> Record;
  auto DeleteRecord(Key key) -> void;

  // Zero-Copy Access to Record Scalar Fields
  // auto GetRecordScalarField(Key key, const std::string& field) const ->
  // Scalar;

  auto PutIndex(const std::string& field, const IvfFlatIndex& index) -> void;
  auto GetIndex(const std::string& field) -> std::unique_ptr<IvfFlatIndex>;
  auto DeleteIndex(const std::string& field) -> void;

  auto GetIterator(std::string_view prefix)
      -> std::unique_ptr<rocksdb::Iterator>;

  static auto MakeRecordKey(Key key) -> std::string;
  static auto MakeIndexKey(const std::string& field) -> std::string;
  static auto MakeCentroidKey(const std::string& field) -> std::string;

  static auto GetKey(rocksdb::Slice rdb_key) -> Key;

  static constexpr const char* kSchemaPrefix = "s:";
  static constexpr const char* kRecordPrefix = "r:";
  static constexpr const char* kIndexPrefix = "i:";
  static constexpr const char* kCentroidPrefix = "c:";

 private:
  std::unique_ptr<rocksdb::DB> db_;
  const DbOptions& options_;
};

}  // namespace rox