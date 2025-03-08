
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "roxdb/db.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  constexpr const char* kUsage = "./main <path-to-fvec>";
  if (argc != 2) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string path = argv[1];
  const size_t num_vectors = 2000;
  const auto vectors = LoadFvecs(path, num_vectors);
  const size_t n_centroids = 32;
  const size_t n_probe = 8;
  const size_t k = 100;

  rox::Schema schema;
  schema.AddVectorField("vec", 128, n_centroids);

  rox::DbOptions options;
  options.ivf_nprobe = n_probe;
  rox::DB db("/tmp/roxdb", options, schema);

  auto clustering_start = std::chrono::high_resolution_clock::now();
  const auto centroids = FindCentroids(vectors, n_centroids);
  auto clustering_end = std::chrono::high_resolution_clock::now();
  std::cout << "Clustering time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   clustering_end - clustering_start)
                   .count()
            << "ms" << std::endl;
  db.SetCentroids("vec", centroids);

  // Print distribution
  PrintClusterDistribution(vectors, centroids, n_centroids);

  // Load vectors into the database
  auto put_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < vectors.size(); ++i) {
    rox::Record record;
    record.vectors.push_back(vectors[i]);
    db.PutRecord(i, record);
  }
  auto put_end = std::chrono::high_resolution_clock::now();
  std::cout << "Put time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(put_end -
                                                                     put_start)
                   .count()
            << "ms" << std::endl;

  // Find k cloest vectors to the first vector
  rox::Query q;
  q.AddVector("vec", vectors[0]);
  q.WithLimit(k);
  // KNN Search
  auto knn_start = std::chrono::high_resolution_clock::now();
  auto results = db.KnnSearch(q);
  auto knn_end = std::chrono::high_resolution_clock::now();
  // Full Scan
  auto full_scan_start = std::chrono::high_resolution_clock::now();
  auto gt = db.FullScan(q);
  auto full_scan_end = std::chrono::high_resolution_clock::now();

  // Print results with distance and assigned cluster
  CompareResults(db, results, gt);

  std::cout << "Recall@" << k << ": " << GetRecallAtK(k, results, gt)
            << std::endl;
  std::cout << "KNN Search time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(knn_end -
                                                                     knn_start)
                   .count()
            << "ms" << std::endl;
  std::cout << "Full Scan time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   full_scan_end - full_scan_start)
                   .count()
            << "ms" << std::endl;

  return 0;
}