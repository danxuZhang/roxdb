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

DB::DB(const std::string &path, const DbOptions &options)
    : impl_(std::make_unique<DbImpl>(path, options)) {}

DB::DB(const std::string &path, const DbOptions &options, const Schema &schema)
    : impl_(std::make_unique<DbImpl>(path, options, schema)) {}

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

auto DB::KnnSearch(const Query &query, size_t nprobe) const
    -> std::vector<QueryResult> {
  return impl_->KnnSearch(query, nprobe);
}

auto DB::FlushRecords() -> void { impl_->FlushRecords(); }

auto DB::KnnSearchIterativeMerge(const Query &query, size_t nprobe,
                                 size_t k_threshold) const
    -> std::vector<QueryResult> {
  return impl_->KnnSearchIterativeMerge(query, nprobe, k_threshold);
}

auto DB::KnnSearchVBase(const Query &query, size_t nprobe, size_t n2)
    -> std::vector<QueryResult> {
  return impl_->KnnSearchVBase(query, nprobe, n2);
}

}  // namespace rox