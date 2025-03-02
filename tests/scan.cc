#include <gtest/gtest.h>
#include <roxdb/db.h>

#include <algorithm>
#include <vector>

TEST(Scan, SingleVectorScan) {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddVectorField("vec", 3, 0);

  rox::DB db("/tmp/roxdb", schema, options);

  // Put random record
  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Vector v = {1.0, 3.0, 5.0};
    std::ranges::for_each(v, [i](auto& x) { x *= i; });
    rox::Record record;
    record.id = i;
    record.vectors.push_back(v);
    db.PutRecord(i, record);
  }

  // Find 3 cloest vectors to (9, 27, 45)
  rox::Query query;
  rox::Vector v = {9.0, 27.0, 45.0};
  query.AddVector("vec", v);
  query.WithLimit(3);

  auto results = db.FullScan(query);
  // for (const auto& r : results) {
  //   std::cout << r.id << " ";
  // }
  // std::cout << std::endl;
  EXPECT_EQ(results.size(), 3);
  EXPECT_EQ(results[0].id, 9);
  EXPECT_EQ(results[1].id, 8);
  EXPECT_EQ(results[2].id, 7);
}

TEST(Scan, SingleVectorScanWithWeight) {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddScalarField("val", rox::ScalarField::Type::kInt)
      .AddVectorField("vec", 3, 0);

  rox::DB db("/tmp/roxdb", schema, options);

  // Put random record
  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Vector v = {1.0, 3.0, 5.0};
    std::ranges::for_each(v, [i](auto& x) { x *= i; });
    rox::Record record;
    record.id = i;
    record.scalars.emplace_back(static_cast<int>(i % 2));
    record.vectors.push_back(v);
    db.PutRecord(i, record);
  }

  // Find 3 cloest vectors to (9, 27, 45)
  rox::Query query;
  rox::Vector v = {9.0, 27.0, 45.0};
  query.AddVector("vec", v, 1.0);
  query.AddScalarFilter("val", rox::ScalarFilter::Op::kEq, 0);
  query.WithLimit(3);

  auto results = db.FullScan(query);
  EXPECT_EQ(results.size(), 3);
  EXPECT_EQ(results[0].id, 8);
  EXPECT_EQ(results[1].id, 6);
  EXPECT_EQ(results[2].id, 4);
}

TEST(Scan, MultiVectorScan) {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddVectorField("vec1", 3, 0).AddVectorField("vec2", 4, 0);

  rox::DB db("/tmp/roxdb", schema, options);

  rox::Vector target1 = {2.0, 4.0, 6.0};
  rox::Vector target2 = {2.0, 4.0, 6.0, 8.0};
  float weight1 = 0.4;
  float weight2 = 0.6;

  rox::Query query;
  query.AddVector("vec1", target1, weight1);
  query.AddVector("vec2", target2, weight2);
  query.WithLimit(3);

  std::vector<rox::Record> records;

  // Generate random records
  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Vector v1 = {1.0, 3.0, 5.0};
    rox::Vector v2 = {1.0, 3.0, 5.0, 7.0};
    std::ranges::for_each(v1, [i](auto& x) { x *= i; });
    std::ranges::for_each(v2, [i](auto& x) { x *= i; });
    rox::Record record;
    record.id = i;
    record.vectors.push_back(v1);
    record.vectors.push_back(v2);
    db.PutRecord(i, record);
    records.push_back(record);
  }

  auto results = db.FullScan(query);
  EXPECT_EQ(results.size(), 3);

  // sort all records
  auto get_distance_l2_sq = [](const rox::Vector& v1, const rox::Vector& v2) {
    float dist = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
      dist += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return dist;
  };
  auto comp = [&](const rox::Record& r1, const rox::Record& r2) {
    float dist1 = (get_distance_l2_sq(r1.vectors[0], target1) * weight1) +
                  (get_distance_l2_sq(r1.vectors[1], target2) * weight2);
    float dist2 = (get_distance_l2_sq(r2.vectors[0], target1) * weight1) +
                  (get_distance_l2_sq(r2.vectors[1], target2) * weight2);
    return dist1 < dist2;
  };
  std::ranges::sort(records, comp);

  // for (const auto &r : results) {
  //   std::cout << r.id << " ";
  // }
  // std::cout << std::endl;

  // for (const auto &r : records) {
  //   std::cout << r.id << " ";
  // }
  // std::cout << std::endl;

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(results[i].id, records[i].id);
  }
}
