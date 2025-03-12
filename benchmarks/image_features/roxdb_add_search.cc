#include <H5Cpp.h>
#include <H5File.h>

#include <chrono>
#include <iostream>
#include <numeric>

#include "io.h"
#include "query.h"
#include "roxdb/db.h"
#include "utils.h"

constexpr bool kEvaluate = true;

int main(int argc, char* argv[]) {
  constexpr const char* kUsage =
      "Usage: roxdb_add_search <db_path> <dataset_path.5>  <query_path.h5>"
      "<sift_centroid.fvecs> <gist_centroid.fvecs>";

  if (argc != 6) {
    std::cerr << kUsage << std::endl;
    return 1;
  }

  const std::string db_path = argv[1];
  const std::string dataset_path = argv[2];
  const std::string sift_centroid_path = argv[3];
  const std::string gist_centroid_path = argv[4];

  PrintHdf5FileInfo(dataset_path);
  H5::H5File file(dataset_path, H5F_ACC_RDONLY);

  Dataset dataset = ReadDataset(file);
  PrintDatasetSummary(dataset);
  Dataset query_dataset = ReadDataset(file);
  PrintDatasetSummary(query_dataset);

  const auto n = dataset.num_records;
  const auto n_clusters = 1000;

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

  // Load Centroids
  FvecsReader sift_centroid_reader(sift_centroid_path);
  std::vector<rox::Vector> sift_centroids;
  sift_centroids.reserve(n_clusters);
  while (sift_centroid_reader.HasNext()) {
    sift_centroids.push_back(sift_centroid_reader.Get());
    sift_centroid_reader.Next();
  }
  FvecsReader gist_centroid_reader(gist_centroid_path);
  std::vector<rox::Vector> gist_centroids;
  gist_centroids.reserve(n_clusters);
  while (gist_centroid_reader.HasNext()) {
    gist_centroids.push_back(gist_centroid_reader.Get());
    gist_centroid_reader.Next();
  }

  db.SetCentroids("sift", sift_centroids);
  db.SetCentroids("gist", gist_centroids);

  // Load dataset
  auto start = std::chrono::high_resolution_clock::now();
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

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Loading time: " << duration.count() << "ms" << std::endl;

  std::cout << "Successfully loaded dataset" << std::endl;

  const auto queries = GetQueries(dataset);
  const auto n_query = queries.size();
  std::vector<std::vector<int64_t>> times(n_query);
  std::vector<std::vector<int64_t>> scan_times(n_query);
  std::vector<std::vector<float>> recalls(n_query);
  for (size_t i = 0; i < kIters; ++i) {
    std::cout << "Iteration " << i + 1 << std::endl;
    for (size_t j = 0; j < queries.size(); ++j) {
      const auto& query = queries[j];
      auto start = std::chrono::high_resolution_clock::now();
      auto results = db.KnnSearch(query, 24);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      times[j].push_back(duration.count());

      if (kEvaluate) {
        start = std::chrono::high_resolution_clock::now();
        auto gt = db.FullScan(query);
        end = std::chrono::high_resolution_clock::now();
        duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        scan_times[j].push_back(duration.count());
        auto recall = GetRecallAtK(query.GetLimit(), results, gt);
        recalls[j].push_back(recall);
      }  // if (kEvaluate)
    }  // for (size_t j = 0; j < queries.size(); ++j)
  }  // for (size_t i = 0; i < kIters; ++i)

  for (size_t i = 0; i < queries.size(); i++) {
    std::cout << "Query " << i + 1 << std::endl;
    std::cout << "Average search time: "
              << std::reduce(times[i].begin(), times[i].end(), 0.0) / kIters
              << "ms" << std::endl;
    if (kEvaluate) {
      std::cout << "Average scan time: "
                << std::reduce(scan_times[i].begin(), scan_times[i].end(),
                               0.0) /
                       kIters
                << "ms" << std::endl;
      std::cout << "Average recall: "
                << std::reduce(recalls[i].begin(), recalls[i].end(), 0.0) /
                       kIters
                << std::endl;
    }  // if (kEvaluate)
  }  // for (size_t i = 0; i < queries.size(); i++)

  return 0;
}