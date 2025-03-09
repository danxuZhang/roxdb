#include <H5Cpp.h>
#include <H5File.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "io.h"
#include "roxdb/db.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  constexpr const char* kUsage = "Usage: roxdb_add <db_path> <dataset_path.5>";

  if (argc != 3) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string db_path = argv[1];
  const std::string dataset_path = argv[2];

  PrintHdf5FileInfo(dataset_path);
  H5::H5File file(dataset_path, H5F_ACC_RDONLY);

  Dataset dataset = ReadDataset(file);
  PrintDatasetSummary(dataset);

  const auto n = dataset.num_records;
  const auto n_clusters = static_cast<size_t>(std::sqrt(n));
  std::cout << "Number of records: " << n << std::endl;
  std::cout << "Number of clusters: " << n_clusters << std::endl;
  rox::Schema schema;
  schema.AddVectorField("sift", dataset.sift_dim, n_clusters);
  schema.AddVectorField("gist", dataset.gist_dim, n_clusters);
  schema.AddScalarField("image_id", rox::ScalarField::Type::kInt);
  schema.AddScalarField("category", rox::ScalarField::Type::kInt);
  schema.AddScalarField("confidence", rox::ScalarField::Type::kDouble);
  schema.AddScalarField("votes", rox::ScalarField::Type::kInt);

  rox::DbOptions options;
  options.create_if_missing = true;
  rox::DB db(db_path, options, schema);

  // Find centroids
  auto start = std::chrono::high_resolution_clock::now();
  auto sift_centroids = FindCentroids(dataset.sift, n_clusters);
  auto gist_centroids = FindCentroids(dataset.gist, n_clusters);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Clustering time: " << duration.count() << "ms" << std::endl;

  db.SetCentroids("sift", sift_centroids);
  db.SetCentroids("gist", gist_centroids);

  // Load dataset
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; i++) {
    rox::Record record;
    record.vectors.push_back(dataset.sift[i]);
    record.vectors.push_back(dataset.gist[i]);
    record.scalars.emplace_back(dataset.image_id[i]);
    record.scalars.emplace_back(dataset.category[i]);
    record.scalars.emplace_back(static_cast<double>(dataset.confidence[i]));
    record.scalars.emplace_back(dataset.votes[i]);
    db.PutRecord(i, record);
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Loading time: " << duration.count() << "ms" << std::endl;

  std::cout << "Successfully loaded dataset" << std::endl;

  return 0;
}