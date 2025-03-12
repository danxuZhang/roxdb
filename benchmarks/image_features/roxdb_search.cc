#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "io.h"
#include "roxdb/db.h"
#include "utils.h"

constexpr const size_t kIters = 10;

namespace {

auto GetQueries(const Dataset& dataset) -> std::vector<rox::Query> {
  const size_t k = 100;

  // Q1: Single KNN on SIFT vectors
  rox::Query q1;
  q1.AddVector("sift", dataset.sift[0]);
  q1.WithLimit(k);

  // Q2: Single KNN on GIST vectors
  rox::Query q2;
  q2.AddVector("gist", dataset.gist[0]);
  q2.WithLimit(k);

  // Q3: Single KNN on SIFT vectors with filters
  rox::Query q3;
  q3.AddVector("sift", dataset.sift[0]);
  q3.AddScalarFilter("category", rox::ScalarFilter::Op::kEq, 5);
  q3.AddScalarFilter("confidence", rox::ScalarFilter::Op::kLt, 0.5);
  q3.WithLimit(k);

  // Q4: Single KNN on GIST vectors with filters
  rox::Query q4;
  q4.AddVector("gist", dataset.gist[0]);
  q4.AddScalarFilter("category", rox::ScalarFilter::Op::kEq, 5);
  q4.AddScalarFilter("confidence", rox::ScalarFilter::Op::kLt, 0.5);
  q4.WithLimit(k);

  // Q5: Multi KNN on SIFT and GIST vectors
  rox::Query q5;
  q5.AddVector("sift", dataset.sift[0]);
  q5.AddVector("gist", dataset.gist[0]);
  q5.WithLimit(k);

  // Q6: Multi KNN on SIFT and GIST vectors with filters
  rox::Query q6;
  q6.AddVector("sift", dataset.sift[0]);
  q6.AddVector("gist", dataset.gist[0]);
  q6.AddScalarFilter("category", rox::ScalarFilter::Op::kEq, 5);
  q6.AddScalarFilter("confidence", rox::ScalarFilter::Op::kLt, 0.5);
  q6.WithLimit(k);

  return {q1, q2, q3, q4, q5, q6};
}

}  // namespace

int main(int argc, char* argv[]) {
  constexpr const char* kUsage =
      "Usage: roxdb_search <db_path> <queries_path.h5> --evaluate";
  if (argc < 3 || argc > 4 ||
      (argc == 4 && std::string(argv[3]) != "--evaluate")) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string db_path = argv[1];
  const std::string dataset_path = argv[2];
  const bool evaluate = argc == 4;
  rox::DbOptions options;
  options.create_if_missing = false;
  rox::DB db(db_path, options);

  PrintHdf5FileInfo(dataset_path);
  H5::H5File file(dataset_path, H5F_ACC_RDONLY);
  Dataset dataset = ReadDataset(file);
  PrintDatasetSummary(dataset);

  const auto queries = GetQueries(dataset);
  const auto n_query = queries.size();
  std::vector<std::vector<int64_t>> times(n_query);
  std::vector<std::vector<int64_t>> scan_times(n_query);
  std::vector<std::vector<float>> recalls(n_query);
  for (size_t i = 0; i < kIters; i++) {
    std::cout << "Iteration " << i + 1 << std::endl;
    for (size_t j = 0; j < n_query; ++j) {
      std::cout << "Query " << j + 1 << std::endl;
      const auto& query = queries[j];
      size_t nprobe = 24;
      auto start = std::chrono::high_resolution_clock::now();
      auto results = db.KnnSearch(query, nprobe);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      times[j].push_back(duration.count());

      if (evaluate) {
        start = std::chrono::high_resolution_clock::now();
        auto gt = db.FullScan(query);
        end = std::chrono::high_resolution_clock::now();
        duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        scan_times[j].push_back(duration.count());
        auto recall = GetRecallAtK(query.GetLimit(), results, gt);
        recalls[j].push_back(recall);
      }
    }
  }  // search loop

  for (size_t i = 0; i < queries.size(); i++) {
    std::cout << "Query " << i + 1 << std::endl;
    std::cout << "Average search time: "
              << std::reduce(times[i].begin(), times[i].end(), 0.0) / kIters
              << "ms" << std::endl;
    if (evaluate) {
      std::cout << "Average scan time: "
                << std::reduce(scan_times[i].begin(), scan_times[i].end(),
                               0.0) /
                       kIters
                << "ms" << std::endl;
      std::cout << "Average recall: "
                << std::reduce(recalls[i].begin(), recalls[i].end(), 0.0) /
                       kIters
                << std::endl;
    }
  }  // print loop

  return 0;
}