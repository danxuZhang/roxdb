#include <chrono>
#include <iostream>

#include "roxdb/db.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  constexpr const char* kUsage = "./load_sift <path-to-db> <path-to-fvec>";

  if (argc != 3) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string db_path = argv[1];
  const std::string fvec_path = argv[2];

  const size_t n_vectors = 10000;  // 10K
  const size_t n_centroids = 100;
  std::vector<rox::Vector> sift;
  sift.reserve(n_vectors);
  {
    FvecsReader reader(fvec_path);
    for (size_t i = 0; i < n_vectors; ++i) {
      sift.push_back(reader.Get());
      reader.Next();
    }
  }

  const auto n = sift.size();
  std::cout << "Loaded " << n << " vectors" << std::endl;

  // Split sift into two vector fields
  const auto vectors1 =
      std::vector<rox::Vector>(sift.begin(), sift.begin() + n / 2);
  const auto vectors2 =
      std::vector<rox::Vector>(sift.begin() + n / 2, sift.end());
  const size_t n_probe = 16;

  rox::Schema schema;
  schema.AddVectorField("vec1", 128, n_centroids);
  schema.AddVectorField("vec2", 128, n_centroids);

  rox::DbOptions options;
  options.ivf_nprobe = n_probe;
  rox::DB db(db_path, options, schema);

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

  return 0;
}