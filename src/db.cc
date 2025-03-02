#include "roxdb/db.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "impl.h"

namespace rox {
auto ScalarToString(const Scalar &scalar) noexcept -> std::string {
  return std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return value;
        } else if constexpr (std::is_same_v<T, double>) {
          return std::to_string(static_cast<double>(value));
        } else if constexpr (std::is_same_v<T, int>) {
          return std::to_string(static_cast<int>(value));
        }
      },
      scalar);
}

auto ScalarFromString(const std::string &str) noexcept -> Scalar {
  if (str.empty()) {
    return std::string();
  }
  if (std::ranges::all_of(str, ::isdigit)) {
    return std::stoi(str);
  }
  try {
    return std::stod(str);
  } catch (const std::invalid_argument &) {
    return str;
  }
}

auto Schema::AddVectorField(const std::string &name, size_t dimension,
                            size_t num_centroids) -> Schema & {
  if (vector_field_idx.contains(name)) {
    throw std::invalid_argument("Vector field already exists");
  }

  vector_fields.push_back({name, dimension, num_centroids});
  vector_field_idx[name] = vector_fields.size() - 1;
  return *this;
}

auto Schema::AddScalarField(const std::string &name, ScalarField::Type type)
    -> Schema & {
  if (scalar_field_idx.contains(name)) {
    throw std::invalid_argument("Scalar field already exists");
  }

  scalar_fields.push_back({name, type});
  scalar_field_idx[name] = scalar_fields.size() - 1;
  return *this;
}

auto Schema::GetVectorField(const std::string &name) const
    -> const VectorField & {
  if (!vector_field_idx.contains(name)) {
    throw std::invalid_argument("Vector field not found");
  }

  return vector_fields[vector_field_idx.at(name)];
}

auto Schema::GetScalarField(const std::string &name) const
    -> const ScalarField & {
  if (!scalar_field_idx.contains(name)) {
    throw std::invalid_argument("Scalar field not found");
  }

  return scalar_fields[scalar_field_idx.at(name)];
}

auto Query::AddVector(const std::string &field, const Vector &vector,
                      Float weight) -> Query & {
  vectors.emplace_back(field, vector, weight);
  return *this;
}

auto Query::AddScalarFilter(const std::string &field, ScalarFilter::Op op,
                            const Scalar &value) -> Query & {
  filters.push_back({field, op, value});
  return *this;
}

auto Query::WithLimit(size_t limit) -> Query & {
  this->limit = limit;
  return *this;
}

auto Query::GetVectors() const noexcept
    -> const std::vector<std::tuple<std::string, Vector, Float>> & {
  return vectors;
}

auto Query::GetFilters() const noexcept -> std::vector<ScalarFilter> {
  return filters;
}

auto Query::GetLimit() const noexcept -> size_t { return limit; }

auto ApplyFilter(const Schema &schema, const Record &record,
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

DB::DB(const std::string &path, const Schema &schema, const DbOptions &options)
    : impl_(std::make_unique<DbImpl>(path, schema, options)) {}

DB::~DB() = default;

auto DB::PutRecord(Key key, const Record &record) -> void {
  impl_->PutRecord(key, record);
}

auto DB::GetRecord(Key key) const -> Record { return impl_->GetRecord(key); }

auto DB::DeleteRecord(Key key) -> void { impl_->DeleteRecord(key); }

auto DB::SetCentroids(const std::string &field,
                      const std::vector<Vector> &centroids) -> void {
  impl_->SetCentroids(field, centroids);
}

auto DB::FullScan(const Query &query) const -> std::vector<QueryResult> {
  return impl_->FullScan(query);
}

auto DB::KnnSearch(const Query &query) const -> std::vector<QueryResult> {
  return impl_->KnnSearch(query);
}

}  // namespace rox