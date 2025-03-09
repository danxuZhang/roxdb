#pragma once

#include <H5Cpp.h>
#include <H5File.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "roxdb/db.h"

struct Dataset {
  std::vector<rox::Vector> sift;  // SIFT vectors (128D)
  std::vector<rox::Vector> gist;  // GIST vectors (960D)
  std::vector<int> image_id;      // Image IDs
  std::vector<int> category;      // Categories
  std::vector<float> confidence;  // Confidence values
  std::vector<int> votes;         // Vote counts
  int num_records;                // Number of records in the dataset
  int sift_dim;                   // Dimension of SIFT vectors
  int gist_dim;                   // Dimension of GIST vectors
};

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

auto PrintHdf5FileInfo(const std::string& filePath) -> void;

template <typename T>
auto ReadAttribute(const H5::H5File& file, const std::string& attrName) -> T;

auto ReadVector(const H5::H5File& file, const std::string& datasetName)
    -> std::vector<rox::Vector>;

auto ReadIntDataset(const H5::H5File& file, const std::string& datasetName)
    -> std::vector<int>;

auto ReadFloatDataset(const H5::H5File& file, const std::string& datasetName)
    -> std::vector<float>;

auto ReadDataset(const H5::H5File& file) -> Dataset;

auto PrintDatasetSummary(const Dataset& dataset) -> void;