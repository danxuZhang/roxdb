#include "roxdb/db.h"

#include <algorithm>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "roxdb/vector.h"

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

auto DB::PutRecord(Key key, const Record &record) -> void {
  if (records_.contains(key)) {
    throw std::invalid_argument("Record already exists");
  }

  records_[key] = record;
}

auto DB::GetRecord(Key key) const -> Record {
  if (!records_.contains(key)) {
    throw std::invalid_argument("Record not found");
  }

  return records_.at(key);
}

auto DB::DeleteRecord(Key key) -> void {
  if (!records_.contains(key)) {
    throw std::invalid_argument("Record not found");
  }

  records_.erase(key);
}

auto DB::FullScan(const Query &query) const -> std::vector<QueryResult> {
  if (query.GetLimit() == 0) {
    return {};
  }

  auto records =
      records_ |  // Filter records based on scalar filters
      std::views::filter([&](const auto &pair) {
        const auto &[key, record] = pair;
        return std::ranges::all_of(query.filters, [&](const auto &filter) {
          return ApplyFilter(schema_, record, filter);
        });
      }) |  // Calculate distance for each record
      std::views::transform([&](const auto &pair) {
        const auto &[key, record] = pair;
        // Calculate aggregate distance
        Float distance = 0.0;
        for (const auto &[field, query_vector, weight] : query.vectors) {
          const auto &record_vector =
              record.vectors[schema_.vector_field_idx.at(field)];
          distance += GetDistanceL2Sq(query_vector, record_vector) * weight;
        }

        return QueryResult{key, distance};
      });

  // Find top k records
  // Create a max heap pq, the top element is the largest
  // top() is the largest, pop() removes the largest in the heap
  // New candidate only needs to compare with the largest in the heap (top)
  std::priority_queue<QueryResult> pq;
  for (const auto &result : records) {
    // pq.emplace(result);
    if (pq.size() < query.limit) {
      pq.push(result);
    } else if (result.distance < pq.top().distance) {
      pq.pop();
      pq.push(result);
    }
  }

  std::vector<QueryResult> results;
  results.reserve(query.GetLimit());
  while (!pq.empty()) {
    results.push_back(pq.top());
    pq.pop();
  }
  // Reverse the results to get the smallest distance first
  std::ranges::reverse(results);
  return results;
}
}  // namespace rox