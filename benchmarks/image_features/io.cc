#include "io.h"

auto PrintHdf5FileInfo(const std::string& filePath) -> void {
  std::cout << "Attempting to open HDF5 file: " << filePath << std::endl;

  try {
    // Open the file
    H5::H5File file(filePath, H5F_ACC_RDONLY);
    std::cout << "Successfully opened HDF5 file" << std::endl;

    // Get basic file info
    std::cout << "\n=== HDF5 File Structure ===" << std::endl;

    // Print the number of objects in the root group
    H5::Group root = file.openGroup("/");
    hsize_t num_objs = 0;
    H5Gget_num_objs(root.getId(), &num_objs);
    std::cout << "Number of objects in root group: " << num_objs << std::endl;

    // List all objects in root
    std::cout << "\nObjects in root group:" << std::endl;
    for (hsize_t i = 0; i < num_objs; i++) {
      char obj_name[256];
      H5Gget_objname_by_idx(root.getId(), i, obj_name, 256);
      H5G_obj_t obj_type = H5Gget_objtype_by_idx(root.getId(), i);

      std::string type_str;
      switch (obj_type) {
        case H5G_GROUP:
          type_str = "Group";
          break;
        case H5G_DATASET:
          type_str = "Dataset";
          break;
        case H5G_TYPE:
          type_str = "Named Datatype";
          break;
        default:
          type_str = "Unknown";
          break;
      }

      std::cout << "  " << obj_name << " (Type: " << type_str << ")"
                << std::endl;
    }

    // List root attributes
    std::cout << "\nAttributes in root group:" << std::endl;
    int num_attrs = H5Aget_num_attrs(root.getId());
    for (int i = 0; i < num_attrs; i++) {
      hid_t attr_id = H5Aopen_idx(root.getId(), static_cast<unsigned int>(i));
      char attr_name[256];
      H5Aget_name(attr_id, 256, attr_name);
      std::cout << "  " << attr_name << std::endl;
      H5Aclose(attr_id);
    }

    std::cout << "========================\n" << std::endl;
  } catch (const H5::Exception& error) {
    std::cout << "ERROR: Failed to open or read HDF5 file" << std::endl;
    H5::Exception::printErrorStack();
  }
}

template <typename T>
auto ReadAttribute(const H5::H5File& file, const std::string& attrName) -> T {
  H5::Attribute attr = file.openAttribute(attrName);
  T value;
  attr.read(attr.getDataType(), &value);
  return value;
}

auto ReadVector(const H5::H5File& file, const std::string& datasetName)
    -> std::vector<rox::Vector> {
  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  hsize_t dims_out[2];
  dataspace.getSimpleExtentDims(dims_out, nullptr);
  const int num_records = dims_out[0];
  const int dim = dims_out[1];

  std::vector<float> buffer(num_records * dim);
  dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

  std::vector<rox::Vector> vectors(num_records, rox::Vector(dim));
  for (int i = 0; i < num_records; i++) {
    for (int j = 0; j < dim; j++) {
      vectors[i][j] = buffer[(i * dim) + j];
    }
  }

  return vectors;
}

auto ReadIntDataset(const H5::H5File& file, const std::string& datasetName)
    -> std::vector<int> {
  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  hsize_t dims_out[1];
  dataspace.getSimpleExtentDims(dims_out, nullptr);
  const int num_records = dims_out[0];

  std::vector<int> values(num_records);
  dataset.read(values.data(), H5::PredType::NATIVE_INT);
  return values;
}

auto ReadFloatDataset(const H5::H5File& file, const std::string& datasetName)
    -> std::vector<float> {
  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  hsize_t dims_out[1];
  dataspace.getSimpleExtentDims(dims_out, nullptr);
  const int num_records = dims_out[0];

  std::vector<float> values(num_records);
  dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
  return values;
}

auto ReadDataset(const H5::H5File& file) -> Dataset {
  Dataset dataset;
  dataset.sift = ReadVector(file, "sift");
  dataset.gist = ReadVector(file, "gist");
  dataset.image_id = ReadIntDataset(file, "image_id");
  dataset.category = ReadIntDataset(file, "category");
  dataset.confidence = ReadFloatDataset(file, "confidence");
  dataset.votes = ReadIntDataset(file, "votes");
  dataset.num_records = ReadAttribute<int>(file, "num_records");
  dataset.sift_dim = ReadAttribute<int>(file, "sift_dim");
  dataset.gist_dim = ReadAttribute<int>(file, "gist_dim");
  return dataset;
}

auto PrintDatasetSummary(const Dataset& dataset) -> void {
  std::cout << "Dataset Summary:" << std::endl;
  std::cout << "Number of records: " << dataset.num_records << std::endl;
  std::cout << "SIFT dimension: " << dataset.sift_dim << std::endl;
  std::cout << "GIST dimension: " << dataset.gist_dim << std::endl;
  std::cout << "Number of SIFT vectors: " << dataset.sift.size() << std::endl;
  std::cout << "Number of GIST vectors: " << dataset.gist.size() << std::endl;

  // Print a sample record
  if (!dataset.sift.empty()) {
    std::cout << "\nSample record (index 0):" << std::endl;
    std::cout << "Image ID: " << dataset.image_id[0] << std::endl;
    std::cout << "Category: " << dataset.category[0] << std::endl;
    std::cout << "Confidence: " << dataset.confidence[0] << std::endl;
    std::cout << "Votes: " << dataset.votes[0] << std::endl;

    std::cout << "First 5 values of SIFT vector: ";
    for (int i = 0; i < 5 && i < dataset.sift_dim; i++) {
      std::cout << dataset.sift[0][i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "First 5 values of GIST vector: ";
    for (int i = 0; i < 5 && i < dataset.gist_dim; i++) {
      std::cout << dataset.gist[0][i] << " ";
    }
    std::cout << "..." << std::endl;
  }
}