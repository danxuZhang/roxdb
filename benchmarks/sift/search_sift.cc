#include <chrono>
#include <iostream>

#include "roxdb/db.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  constexpr const char* kUsage = "./sift_sift <path-to-db> <path-to-fvec>";

  if (argc != 3) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string db_path = argv[1];
  const std::string fvec_path = argv[2];
  const size_t k = 50;

  rox::DbOptions options;
  options.create_if_missing = false;
  options.ivf_nprobe = 24;
  rox::DB db(db_path, options);

  const size_t n_query = 10;
  std::vector<rox::Vector> queries;
  queries.reserve(n_query);
  {
    FvecsReader reader(fvec_path);
    // Load 2x n_queries for multi-vector search
    for (size_t i = 0; i < n_query * 2; ++i) {
      reader.Next();
      queries.push_back(reader.Get());
    }
  }

  for (size_t i = 0; i < n_query; ++i) {
    rox::Query query;
    query.limit = 10;
    query.AddVector("vec1", queries[i], 0.6);
    query.AddVector("vec2", queries[i + n_query], 0.4);
    query.WithLimit(k);

    // Run KNN search
    auto start = std::chrono::high_resolution_clock::now();
    const auto results = db.KnnSearch(query);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Query " << i << " time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;

    // Run full scan for comparison
    auto full_scan_start = std::chrono::high_resolution_clock::now();
    const auto full_scan_results = db.FullScan(query);
    auto full_scan_end = std::chrono::high_resolution_clock::now();

    std::cout << "Full scan time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     full_scan_end - full_scan_start)
                     .count()
              << "ms" << std::endl;

    // Calculate Recall
    const auto recall = GetRecallAtK(k, results, full_scan_results);
    std::cout << "Recall@" << k << ": " << recall << std::endl;

    // Compare results
    // CompareResults(db, results, full_scan_results);
  }

  return 0;
}