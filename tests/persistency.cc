#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "roxdb/db.h"

TEST(Persistency, ScalarPersistency) {
  constexpr const char* kPath = "/tmp/roxdb";
  if (std::filesystem::exists(kPath)) {
    std::filesystem::remove_all(kPath);
  }

  {
    rox::DbOptions options;
    options.create_if_missing = true;
    rox::Schema schema;
    schema.AddScalarField("int", rox::ScalarField::Type::kInt)
        .AddScalarField("double", rox::ScalarField::Type::kDouble)
        .AddScalarField("string", rox::ScalarField::Type::kString);

    rox::DB db(kPath, options, schema);

    // Put random record
    const size_t n_records = 10;
    for (size_t i = 0; i < n_records; ++i) {
      rox::Record record;
      record.id = i;
      record.scalars.emplace_back(static_cast<int>(i));
      record.scalars.emplace_back(static_cast<double>(i) * 0.1);
      record.scalars.emplace_back(std::to_string(i));
      db.PutRecord(i, record);
    }

    // Get records
    for (size_t i = 0; i < n_records; ++i) {
      auto record = db.GetRecord(i);
      EXPECT_EQ(std::get<int>(record.scalars[0]), i);
      EXPECT_EQ(std::get<double>(record.scalars[1]), i * 0.1);
      EXPECT_EQ(std::get<std::string>(record.scalars[2]), std::to_string(i));
    }
  }

  {
    rox::DbOptions options;
    options.create_if_missing = false;
    rox::DB db(kPath, options);

    // Get records
    const size_t n_records = 10;
    for (size_t i = 0; i < n_records; ++i) {
      auto record = db.GetRecord(i);
      EXPECT_EQ(std::get<int>(record.scalars[0]), i);
      EXPECT_EQ(std::get<double>(record.scalars[1]), i * 0.1);
      EXPECT_EQ(std::get<std::string>(record.scalars[2]), std::to_string(i));
    }
  }

  std::filesystem::remove_all(kPath);
}

TEST(Persistency, VectorPersistency) {
  constexpr const char* kPath = "/tmp/roxdb";
  if (std::filesystem::exists(kPath)) {
    std::filesystem::remove_all(kPath);
  }

  rox::Schema schema;
  schema.AddVectorField("vec1", 3, 1);
  schema.AddVectorField("vec2", 4, 1);
  schema.AddVectorField("vec3", 5, 1);

  auto vector_equal = [](const auto& v1, const auto& v2) {
    return std::ranges::equal(v1, v2);
  };

  {
    rox::DbOptions options;
    options.create_if_missing = true;

    rox::DB db(kPath, options, schema);

    rox::Vector centroid1 = {1.0, 3.0, 5.0};
    rox::Vector centroid2 = {2.0, 4.0, 6.0, 8.0};
    rox::Vector centroid3 = {3.0, 5.0, 7.0, 9.0, 11.0};
    db.SetCentroids("vec1", {centroid1});
    db.SetCentroids("vec2", {centroid2});
    db.SetCentroids("vec3", {centroid3});

    // Put random record
    const size_t n_records = 10;
    for (size_t i = 0; i < n_records; ++i) {
      rox::Record record;
      record.id = i;
      record.vectors.push_back({1.0, 3.0, 5.0});
      record.vectors.push_back({2.0, 4.0, 6.0, 8.0});
      record.vectors.push_back({3.0, 5.0, 7.0, 9.0, 11.0});
      db.PutRecord(i, record);
    }

    // Get records
    for (size_t i = 0; i < n_records; ++i) {
      auto record = db.GetRecord(i);
      EXPECT_TRUE(vector_equal(record.vectors[0], std::vector{1.0, 3.0, 5.0}));
      EXPECT_TRUE(
          vector_equal(record.vectors[1], std::vector{2.0, 4.0, 6.0, 8.0}));
      EXPECT_TRUE(vector_equal(record.vectors[2],
                               std::vector{3.0, 5.0, 7.0, 9.0, 11.0}));
    }
  }

  {
    rox::DbOptions options;
    options.create_if_missing = false;
    rox::DB db(kPath, options, schema);

    // Get records
    const size_t n_records = 10;
    for (size_t i = 0; i < n_records; ++i) {
      auto record = db.GetRecord(i);
      EXPECT_TRUE(vector_equal(record.vectors[0], std::vector{1.0, 3.0, 5.0}));
      EXPECT_TRUE(
          vector_equal(record.vectors[1], std::vector{2.0, 4.0, 6.0, 8.0}));
      EXPECT_TRUE(vector_equal(record.vectors[2],
                               std::vector{3.0, 5.0, 7.0, 9.0, 11.0}));
    }
  }

  std::filesystem::remove_all(kPath);
}