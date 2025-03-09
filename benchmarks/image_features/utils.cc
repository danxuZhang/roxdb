#include "utils.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <cassert>
#include <iostream>
#include <unordered_set>

using rox::Float;
using rox::Vector;

auto FindCentroids(const std::vector<Vector>& vectors, size_t num_centroids)
    -> std::vector<Vector> {
  int n = vectors.size();
  int d = vectors[0].size();

  std::vector<float> flat_data(n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      flat_data[(i * d) + j] = vectors[i][j];
    }
  }

  faiss::ClusteringParameters cp;
  cp.niter = 25;
  cp.nredo = 5;

  faiss::Clustering kmeans(d, num_centroids, cp);
  faiss::IndexFlatL2 index(d);

  kmeans.train(n, flat_data.data(), index);
  std::vector<std::vector<float>> centroids(num_centroids,
                                            std::vector<float>(d));
  if (kmeans.centroids.size() == num_centroids * d) {
    for (int i = 0; i < num_centroids; i++) {
      for (int j = 0; j < d; j++) {
        centroids[i][j] = kmeans.centroids[(i * d) + j];
      }
    }
  } else {
    std::cerr << "Error: Unexpected centroids size after clustering"
              << std::endl;
  }

  return centroids;
}

auto GetDistanceL2Sq(const Vector& a, const Vector& b) -> Float {
  Float distance = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    distance += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return distance;
}

auto AssignCentroid(const Vector& v, const std::vector<Vector>& centroids,
                    size_t dim) -> size_t {
  assert(!centroids.empty());
  assert(v.size() == dim);
  size_t best_cluster = 0;
  Float best_distance = GetDistanceL2Sq(v, centroids[0]);
  for (size_t cluster = 1; cluster < centroids.size(); ++cluster) {
    const Float distance = GetDistanceL2Sq(v, centroids[cluster]);
    if (distance < best_distance) {
      best_cluster = cluster;
      best_distance = distance;
    }
  }
  return best_cluster;
}

auto GetRecallAtK(size_t k, const std::vector<rox::QueryResult>& results,
                  const std::vector<rox::QueryResult>& gt) -> Float {
  std::unordered_set<rox::Key> gt_keys;
  for (const auto& r : gt) {
    gt_keys.insert(r.id);
  }

  size_t num_retrieved = 0;
  for (size_t i = 0; i < k; ++i) {
    if (gt_keys.contains(results[i].id)) {
      num_retrieved++;
    }
  }

  return static_cast<Float>(num_retrieved) / gt.size();
}

auto PrintClusterDistribution(const std::vector<rox::Vector>& vectors,
                              const std::vector<rox::Vector>& centroids,
                              size_t n_centroids) -> void {
  std::vector<size_t> vectors_per_cluster(n_centroids, 0);
  for (const auto& vec : vectors) {
    const auto cluster_id = AssignCentroid(vec, centroids, 128);
    vectors_per_cluster[cluster_id]++;
  }
  std::cout << "Cluster distribution:" << std::endl;
  size_t empty_clusters = 0;
  for (size_t i = 0; i < n_centroids; ++i) {
    std::cout << "Cluster " << i << ": " << vectors_per_cluster[i] << " vectors"
              << std::endl;
    if (vectors_per_cluster[i] == 0) {
      empty_clusters++;
    }
  }
  std::cout << "Number of empty clusters: " << empty_clusters << std::endl;
}

auto CompareResults(const rox::DB& db,
                    const std::vector<rox::QueryResult>& results,
                    const std::vector<rox::QueryResult>& gt) -> void {
  assert(results.size() == gt.size());
  std::cout << "Comparing results..." << std::endl;
  auto k = results.size();
  std::cout << "Found  " << results.size() << " results" << std::endl;
  for (auto i = 0; i < k; ++i) {
    auto result = results[i];
    auto gt_result = gt[i];
    auto record = db.GetRecord(result.id);
    auto gt_record = db.GetRecord(gt_result.id);
    std::cout << "Result " << i << ": " << result.id
              << " (distance: " << result.distance << ")"
              << "\t\t\t";
    std::cout << "GT " << i << ": " << gt_result.id
              << " (distance: " << gt_result.distance << ")" << std::endl;
  }
}
