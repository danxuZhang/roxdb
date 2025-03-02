#include <gtest/gtest.h>

#include <algorithm>

#include "roxdb/db.h"

TEST(CRUD, ScalarPutGet) {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddScalarField("name", rox::ScalarField::Type::kString)
      .AddScalarField("age", rox::ScalarField::Type::kInt)
      .AddScalarField("height", rox::ScalarField::Type::kDouble);

  rox::DB db("/tmp/roxdb", schema, options);

  // Put random record
  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Record record;
    record.id = i;
    record.scalars.emplace_back("Alice" + std::to_string(i));
    record.scalars.emplace_back(static_cast<int>(20 + i));
    record.scalars.emplace_back(160.0 + static_cast<double>(i));
    db.PutRecord(i, record);
  }

  // Get and check records
  for (size_t i = 0; i < n_records; ++i) {
    auto record = db.GetRecord(i);
    EXPECT_EQ(std::get<std::string>(record.scalars[0]),
              "Alice" + std::to_string(i));
    EXPECT_EQ(std::get<int>(record.scalars[1]), 20 + i);
    EXPECT_EQ(std::get<double>(record.scalars[2]),
              160.0 + static_cast<double>(i));
  }
}

TEST(CRUD, HybridPutGet) {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddScalarField("name", rox::ScalarField::Type::kString)
      .AddScalarField("age", rox::ScalarField::Type::kInt)
      .AddVectorField("v1", 3, 0)
      .AddVectorField("v2", 4, 0);

  rox::DB db("/tmp/roxdb", schema, options);

  // Put random record
  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Vector v1 = {1.0, 3.0, 5.0};
    std::ranges::for_each(v1, [i](auto &x) { x *= i; });
    rox::Vector v2 = {2.0, 4.0, 6.0, 8.0};
    std::ranges::for_each(v2, [i](auto &x) { x *= i; });
    rox::Record record;
    record.id = i;
    record.scalars.emplace_back("Alice" + std::to_string(i));
    record.scalars.emplace_back(static_cast<int>(20 + i));
    record.vectors.push_back(v1);
    record.vectors.push_back(v2);
    db.PutRecord(i, record);
  }

  // Get and check records
  for (size_t i = 0; i < n_records; ++i) {
    auto record = db.GetRecord(i);
    EXPECT_EQ(std::get<std::string>(record.scalars[0]),
              "Alice" + std::to_string(i));
    EXPECT_EQ(std::get<int>(record.scalars[1]), 20 + i);
    EXPECT_EQ(record.vectors[0][0], 1.0 * i);
    EXPECT_EQ(record.vectors[0][1], 3.0 * i);
    EXPECT_EQ(record.vectors[0][2], 5.0 * i);
    EXPECT_EQ(record.vectors[1][0], 2.0 * i);
    EXPECT_EQ(record.vectors[1][1], 4.0 * i);
    EXPECT_EQ(record.vectors[1][2], 6.0 * i);
    EXPECT_EQ(record.vectors[1][3], 8.0 * i);
  }
}

TEST(CRUD, Delete) {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddScalarField("name", rox::ScalarField::Type::kString)
      .AddScalarField("age", rox::ScalarField::Type::kInt)
      .AddScalarField("height", rox::ScalarField::Type::kDouble)
      .AddVectorField("vec", 4, 0);

  rox::DB db("/tmp/roxdb", schema, options);

  // Put random record
  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Vector v = {1.0, 3.0, 5.0, 7.0};
    std::ranges::for_each(v, [i](auto &x) { x *= i; });
    rox::Record record;
    record.id = i;
    record.scalars.emplace_back("Alice" + std::to_string(i));
    record.scalars.emplace_back(static_cast<int>(20 + i));
    record.scalars.emplace_back(160.0 + static_cast<double>(i));
    record.vectors.push_back(v);
    db.PutRecord(i, record);
  }

  // Get and Delete records
  for (size_t i = 0; i < n_records; ++i) {
    auto record = db.GetRecord(i);
    EXPECT_EQ(std::get<std::string>(record.scalars[0]),
              "Alice" + std::to_string(i));
    EXPECT_EQ(std::get<int>(record.scalars[1]), 20 + i);
    EXPECT_EQ(std::get<double>(record.scalars[2]),
              160.0 + static_cast<double>(i));
    EXPECT_EQ(record.vectors[0][0], 1.0 * i);
    EXPECT_EQ(record.vectors[0][1], 3.0 * i);
    EXPECT_EQ(record.vectors[0][2], 5.0 * i);
    EXPECT_EQ(record.vectors[0][3], 7.0 * i);
    db.DeleteRecord(i);
    EXPECT_THROW(db.GetRecord(i), std::invalid_argument);
  }
}