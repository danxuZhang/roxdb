#include <chrono>
#include <iostream>
#include <string>

#include "roxdb/db.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  constexpr const char* kUsage =
      "./multi_vector <path-to-fvec1> <path-to-fvec2>";
  if (argc != 3) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string path1 = argv[1];
  const std::string path2 = argv[2];
  const size_t num_vectors = 2000;
  const auto vectors1 = LoadFvecs(path1, num_vectors);
  const auto vectors2 = LoadFvecs(path2, num_vectors);
  const size_t n_centroids = 32;
  const size_t n_probe = 8;
  const size_t k = 100;

  rox::Schema schema;
  schema.AddVectorField("vec1", 128, n_centroids);
  schema.AddVectorField("vec2", 128, n_centroids);

  rox::DbOptions options;
  options.ivf_nprobe = n_probe;
  rox::DB db("/tmp/roxdb", schema, options);

  auto clustering_start = std::chrono::high_resolution_clock::now();
  const auto centroids1 = FindCentroids(vectors1, n_centroids);
  const auto centroids2 = FindCentroids(vectors2, n_centroids);
  auto clustering_end = std::chrono::high_resolution_clock::now();
  std::cout << "Clustering time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   clustering_end - clustering_start)
                   .count()
            << "ms" << std::endl;
  db.SetCentroids("vec1", centroids1);
  db.SetCentroids("vec2", centroids2);

  auto put_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < vectors1.size(); ++i) {
    rox::Record record;
    record.vectors.push_back(vectors1[i]);
    record.vectors.push_back(vectors2[i]);
    db.PutRecord(i, record);
  }
  auto put_end = std::chrono::high_resolution_clock::now();
  std::cout << "Put time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(put_end -
                                                                     put_start)
                   .count()
            << "ms" << std::endl;

  rox::Query q;
  q.AddVector("vec1", vectors1[0], 0.7);
  q.AddVector("vec2", vectors2[0], 0.3);
  q.WithLimit(k);

  auto knn_start = std::chrono::high_resolution_clock::now();
  auto results = db.KnnSearch(q);
  auto knn_end = std::chrono::high_resolution_clock::now();

  auto full_scan_start = std::chrono::high_resolution_clock::now();
  auto gt = db.FullScan(q);
  auto full_scan_end = std::chrono::high_resolution_clock::now();

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