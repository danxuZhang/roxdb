#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rox {

struct DbOptions {
  bool create_if_missing = true;
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

  bool operator<(const QueryResult &other) const {
    return distance < other.distance;
  }

  auto operator==(const QueryResult &other) const noexcept -> bool {
    return distance == other.distance;
  }

  auto operator<=>(const QueryResult &other) const {
    // Only compare distance
    return distance <=> other.distance;
  }
};  // struct QueryResult

inline auto ApplyFilter(const Schema &schema, const Record &record,
                        const ScalarFilter &filter) noexcept -> bool {
  const auto &scalar = record.scalars[schema.scalar_field_idx.at(filter.field)];
  switch (filter.op) {
    case ScalarFilter::Op::kEq:
      return scalar == filter.value;
    case ScalarFilter::Op::kNe:
      return scalar != filter.value;
    case ScalarFilter::Op::kGt:
      return scalar > filter.value;
    case ScalarFilter::Op::kGe:
      return scalar >= filter.value;
    case ScalarFilter::Op::kLt:
      return scalar < filter.value;
    case ScalarFilter::Op::kLe:
      return scalar <= filter.value;
  }
  return false;
}

class DbImpl;

class DB {
  constexpr static const char *kVersion = "0.1.0";

 public:
  static auto GetVersion() noexcept -> std::string_view { return kVersion; }

  explicit DB(const std::string &path, const DbOptions &options);
  explicit DB(const std::string &path, const DbOptions &options,
              const Schema &schema);
  ~DB();
  DB(const DB &) = delete;             // non-copyable
  DB &operator=(const DB &) = delete;  // non-assignable

  auto PutRecord(Key key, const Record &record) -> void;
  auto GetRecord(Key key) const -> Record;
  auto DeleteRecord(Key key) -> void;
  auto FlushRecords() -> void;

  auto SetCentroids(const std::string &field,
                    const std::vector<Vector> &centroids) -> void;

  auto FullScan(const Query &query) const -> std::vector<QueryResult>;
  auto KnnSearch(const Query &query, size_t nprobe = 1) const
      -> std::vector<QueryResult>;

  auto KnnSearchIterativeMerge(const Query &query, size_t nprobe,
                               size_t k_threshold) const
      -> std::vector<QueryResult>;

  auto KnnSearchVBase(const Query &query, size_t nprobe, size_t n2)
      -> std::vector<QueryResult>;

 private:
  std::unique_ptr<DbImpl> impl_;

};  // class DB

}  // namespace rox