#include <algorithm>
#include <filesystem>
#include <iostream>

#include "roxdb/db.h"

int main() {
  rox::DbOptions options;
  options.create_if_missing = true;
  rox::Schema schema;
  schema.AddVectorField("vec", 3, 1);

  if (std::filesystem::exists("/tmp/roxdb")) {
    std::filesystem::remove_all("/tmp/roxdb");
  }
  rox::DB db("/tmp/roxdb", options, schema);

  // Random centroid, unused
  rox::Vector centroid = {1.0, 3.0, 5.0};
  db.SetCentroids("vec", {centroid});

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
  for (const auto& r : results) {
    std::cout << r.id << " ";
  }
  std::cout << std::endl;
}