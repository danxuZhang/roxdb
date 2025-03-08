#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "roxdb/db.h"

TEST(KNN, SingleVector) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<rox::Float> dist(-0.1, 0.1);

  rox::Schema schema;
  schema.AddVectorField("vec", 2, 4);

  rox::DbOptions options;
  rox::DB db("/tmp/roxdb", options, schema);

  const rox::Vector c0 = {0, 0};
  const rox::Vector c1 = {0, 1};
  const rox::Vector c2 = {1, 0};
  const rox::Vector c3 = {1, 1};
  const std::vector<rox::Vector> centroids = {c0, c1, c2, c3};
  db.SetCentroids("vec", centroids);

  // Put vectors with random offset around centroids
  const size_t n_records = 16;
  for (size_t i = 0; i < n_records; ++i) {
    const auto &centroid = centroids[i % 4];
    const rox::Vector v = {centroid[0] + dist(gen), centroid[1] + dist(gen)};
    rox::Record record;
    record.id = i;
    record.vectors.push_back(v);
    db.PutRecord(i, record);
  }

  // Find 3 cloest vectors to (0, 0)
  rox::Query q1;
  rox::Vector v1 = {0.0, 0.0};
  q1.AddVector("vec", v1);
  q1.WithLimit(3);
  auto results = db.KnnSearch(q1);
  auto gt = db.FullScan(q1);
  EXPECT_EQ(results.size(), 3);
  EXPECT_EQ(results[0].id, gt[0].id);
  EXPECT_EQ(results[1].id, gt[1].id);
  EXPECT_EQ(results[2].id, gt[2].id);

  // // Find 3 cloest vectors to (1, 1)
  rox::Query q2;
  rox::Vector v2 = {1.0, 1.0};
  q2.AddVector("vec", v2);
  q2.WithLimit(3);
  results = db.KnnSearch(q2);
  gt = db.FullScan(q2);
  EXPECT_EQ(results.size(), 3);
  EXPECT_EQ(results[0].id, gt[0].id);
  EXPECT_EQ(results[1].id, gt[1].id);
  EXPECT_EQ(results[2].id, gt[2].id);
}

TEST(KNN, SingleVectorWFilter) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<rox::Float> dist(-0.1, 0.1);

  rox::Schema schema;
  schema.AddVectorField("vec", 2, 4);
  schema.AddScalarField("idx", rox::ScalarField::Type::kInt);

  rox::DbOptions options;
  options.ivf_nprobe = 3;
  rox::DB db("/tmp/roxdb", options, schema);

  const rox::Vector c0 = {0, 0};
  const rox::Vector c1 = {0, 1};
  const rox::Vector c2 = {1, 0};
  const rox::Vector c3 = {1, 1};
  const std::vector<rox::Vector> centroids = {c0, c1, c2, c3};
  db.SetCentroids("vec", centroids);

  // Put vectors with random offset around centroids
  const size_t n_records = 16;
  for (size_t i = 0; i < n_records; ++i) {
    const auto &centroid = centroids[i % 4];
    const rox::Vector v = {centroid[0] + dist(gen), centroid[1] + dist(gen)};
    rox::Record record;
    record.id = i;
    record.vectors.push_back(v);
    record.scalars.emplace_back(static_cast<int>(i % 2));
    db.PutRecord(i, record);
  }

  // Find 3 cloest vectors to (0, 0)
  rox::Query q1;
  rox::Vector v1 = {0.0, 0.0};
  q1.AddVector("vec", v1);
  q1.AddScalarFilter("idx", rox::ScalarFilter::Op::kEq, 0);
  q1.WithLimit(2);
  auto results = db.KnnSearch(q1);
  auto gt = db.FullScan(q1);
  EXPECT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].id, gt[0].id);
  EXPECT_EQ(results[1].id, gt[1].id);

  // // Find 3 cloest vectors to (1, 1)
  rox::Query q2;
  rox::Vector v2 = {1.0, 1.0};
  q2.AddVector("vec", v2);
  q2.AddScalarFilter("idx", rox::ScalarFilter::Op::kEq, 1);
  q2.WithLimit(2);
  results = db.KnnSearch(q2);
  gt = db.FullScan(q2);
  EXPECT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].id, gt[0].id);
  EXPECT_EQ(results[1].id, gt[1].id);
}
