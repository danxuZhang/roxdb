#include <algorithm>
#include <cassert>
#include <iostream>

#include "roxdb/db.h"

namespace {

auto DbWrite(const std::string& path) {
  rox::Schema schema;
  schema.AddScalarField("name", rox::ScalarField::Type::kString)
      .AddScalarField("age", rox::ScalarField::Type::kInt)
      .AddVectorField("vec", 128, 1);

  rox::DbOptions options;
  options.create_if_missing = true;
  rox::DB db(path, options, schema);

  db.SetCentroids("vec", {rox::Vector(128, 0.0F)});

  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    rox::Record record;
    record.id = i;
    record.scalars = {"name" + std::to_string(i), static_cast<int>(i)};
    record.vectors = {rox::Vector(128, 1.0F * i)};

    db.PutRecord(i, record);
  }

  std::cout << "Wrote " << n_records << " records to " << path << std::endl;
}

auto DbRead(const std::string& path) {
  rox::DbOptions options;
  options.create_if_missing = false;
  rox::DB db(path, options);

  const size_t n_records = 10;
  for (size_t i = 0; i < n_records; ++i) {
    const auto record = db.GetRecord(i);
    std::cout << "Record " << i << ": ";
    for (const auto& scalar : record.scalars) {
      std::cout << rox::ScalarToString(scalar) << " ";
    }
    // assert only one vector
    assert(record.vectors.size() == 1);
    // assert all elements are equal to i
    assert(std::all_of(record.vectors.front().begin(),
                       record.vectors.front().end(),
                       [i](const auto& val) { return val == i; }));

    std::cout << std::endl;
  }

  std::cout << "Read " << n_records << " records from " << path << std::endl;
}
}  // namespace

int main(int argc, char* argv[]) {
  constexpr const char* kUsage = "Usage: read_write read/write <db_path>";
  if (argc != 3) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string mode = argv[1];
  if (mode != "read" && mode != "write") {
    std::cerr << kUsage << std::endl;
    return 1;
  }
  const std::string db_path = argv[2];

  if (mode == "write") {
    DbWrite(db_path);
  } else {
    DbRead(db_path);
  }
}