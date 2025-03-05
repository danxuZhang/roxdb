#pragma once

#include <string>
#include <vector>

#include "roxdb/db.h"

auto LoadFvecs(const std::string& path, int num_vectors)
    -> std::vector<rox::Vector>;

auto FindCentroids(const std::vector<rox::Vector>& vectors,
                   size_t num_centroids) -> std::vector<rox::Vector>;

auto GetDistanceL2Sq(const rox::Vector& a, const rox::Vector& b) -> rox::Float;

auto AssignCentroid(const rox::Vector& v,
                    const std::vector<rox::Vector>& centroids, size_t dim)
    -> size_t;

auto GetRecallAtK(size_t k, const std::vector<rox::QueryResult>& results,
                  const std::vector<rox::QueryResult>& gt) -> rox::Float;

auto PrintClusterDistribution(const std::vector<rox::Vector>& vectors,
                              const std::vector<rox::Vector>& centroids,
                              size_t n_centroids) -> void;

auto CompareResults(const rox::DB& db,
                    const std::vector<rox::QueryResult>& results,
                    const std::vector<rox::QueryResult>& gt) -> void;