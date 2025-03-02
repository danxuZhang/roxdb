#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rox {

struct DbOptions {
  bool create_if_missing = true;
  size_t ivf_n_probe = 1;
};  // struct DbOptions

using Key = uint64_t;
using Float = float;
using Vector = std::vector<Float>;
using Scalar = std::variant<double, int, std::string>;

auto ScalarToString(const Scalar &scalar) noexcept -> std::string;
auto ScalarFromString(const std::string &str) noexcept -> Scalar;

struct VectorField {
  std::string name;
  size_t dim;
  size_t num_centroids;
};  // struct VectorField

struct ScalarField {
  std::string name;
  enum class Type { kDouble, kString, kInt } type;
};  // struct ScalarField

struct ScalarFilter {
  std::string field;
  enum class Op { kEq, kNe, kGt, kGe, kLt, kLe } op;
  Scalar value;
};  // struct ScalarFilter

struct Record {
  Key id;
  std::vector<Scalar> scalars;
  std::vector<Vector> vectors;
};  // struct Record

struct Schema {
  std::vector<VectorField> vector_fields;
  std::vector<ScalarField> scalar_fields;
  std::unordered_map<std::string, size_t> vector_field_idx;
  std::unordered_map<std::string, size_t> scalar_field_idx;

  auto AddVectorField(const std::string &name, size_t dimension,
                      size_t num_centroids) -> Schema &;
  auto AddScalarField(const std::string &name, ScalarField::Type type)
      -> Schema &;

  auto GetVectorField(const std::string &name) const -> const VectorField &;
  auto GetScalarField(const std::string &name) const -> const ScalarField &;
};  // struct Schema

struct Query {
  size_t limit = 0;
  std::vector<std::tuple<std::string, Vector, Float>>
      vectors;  // field_name, vector, weight
  std::vector<ScalarFilter> filters;

  auto AddVector(const std::string &field, const Vector &vector,
                 Float weight = 1.0) -> Query &;
  auto AddScalarFilter(const std::string &field, ScalarFilter::Op op,
                       const Scalar &value) -> Query &;
  auto WithLimit(size_t limit) -> Query &;

  auto GetVectors() const noexcept
      -> const std::vector<std::tuple<std::string, Vector, Float>> &;
  auto GetFilters() const noexcept -> std::vector<ScalarFilter>;
  auto GetLimit() const noexcept -> size_t;
};  // struct Query

struct QueryResult {
  Key id;
  Float distance;

  auto operator==(const QueryResult &other) const noexcept -> bool {
    return distance == other.distance;
  }

  auto operator<=>(const QueryResult &other) const {
    // Only compare distance
    return distance <=> other.distance;
  }
};  // struct QueryResult

auto ApplyFilter(const Schema &schema, const Record &record,
                 const ScalarFilter &filter) noexcept -> bool;

class DB {
  constexpr static const char *kVersion = "0.1.0";

 public:
  static auto GetVersion() noexcept -> std::string_view { return kVersion; }

  DB(const std::string &path, const Schema &schema,
     const DbOptions &options) noexcept
      : path_(path), schema_(schema), options_(options) {};
  ~DB() = default;
  DB(const DB &) = delete;  // non-copyable

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

};  // class DB

}  // namespace rox