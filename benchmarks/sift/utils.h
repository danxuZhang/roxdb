#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "roxdb/db.h"

class FvecsReader {
 public:
  explicit FvecsReader(const std::string& path)
      : path_(path), file_(path, std::ios::binary) {
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open file: " + path);
    }
    ReadNextLine();
  }

  ~FvecsReader() { file_.close(); }

  auto HasNext() const -> bool { return has_next_line_; }
  auto Next() -> void {
    if (!has_next_line_) {
      return;  // Already at EOF or in error state
    }
    ReadNextLine();
  }
  auto Get() const -> const rox::Vector& {
    if (!has_next_line_) {
      throw std::runtime_error("No more lines to read");
    }
    return vector_;
  }
  auto Reset() -> void {
    file_.clear();
    file_.seekg(0, std::ios::beg);
    has_next_line_ = true;
    ReadNextLine();
  }

 private:
  bool has_next_line_ = true;
  rox::Vector vector_;
  std::string path_;
  std::ifstream file_;

  auto ReadNextLine() -> void {
    if (file_.eof()) {
      has_next_line_ = false;
      return;
    }

    int dimension;
    file_.read(reinterpret_cast<char*>(&dimension), sizeof(int));
    if (file_.eof() || file_.fail()) {
      has_next_line_ = false;
      return;
    }

    if (dimension != 128) {
      throw std::runtime_error("Expected 128-dimensional SIFT vector, got " +
                               std::to_string(dimension));
    }

    vector_.resize(dimension);
    file_.read(reinterpret_cast<char*>(vector_.data()),
               dimension * sizeof(float));
    if (file_.fail() && !file_.eof()) {
      has_next_line_ = false;
      throw std::runtime_error("Failed to read vector data");
    }
  }
};  // class FvecsReader

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