#include "storage.h"

#include <memory>
#include <string>

#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers_generated.h"
#include "rocksdb/db.h"
#include "roxdb/db.h"
#include "vector.h"

namespace rox {

RdbStorage::RdbStorage(std::string_view path, const DbOptions& options)
    : options_(options) {
  rocksdb::Options db_options;
  db_options.create_if_missing = options.create_if_missing;

  rocksdb::DB* db_ptr = nullptr;
  rocksdb::Status status =
      rocksdb::DB::Open(db_options, std::string(path), &db_ptr);
  if (status.ok()) {
    db_.reset(db_ptr);
  } else {
    throw std::runtime_error(status.ToString());
  }
}

RdbStorage::~RdbStorage() { db_->Close(); }

auto RdbStorage::GetIterator(std::string_view prefix)
    -> std::unique_ptr<rocksdb::Iterator> {
  auto ptr = std::unique_ptr<rocksdb::Iterator>(
      db_->NewIterator(rocksdb::ReadOptions()));
  ptr->Seek(prefix);
  return ptr;
}

auto RdbStorage::MakeRecordKey(Key key) -> std::string {
  std::stringstream ss;
  ss << kRecordPrefix << key;
  return ss.str();
}

auto RdbStorage::MakeIndexKey(const std::string& field) -> std::string {
  return std::string(kIndexPrefix) + field;
}

auto RdbStorage::MakeCentroidKey(const std::string& field) -> std::string {
  return std::string(kCentroidPrefix) + field;
}

auto RdbStorage::GetKey(rocksdb::Slice rdb_key) -> Key {
  std::string_view key(rdb_key.data(), rdb_key.size());
  if (key.size() <= 2) {
    throw std::invalid_argument("Invalid key");
  }
  return std::stoull(std::string(key.substr(2)));
}

auto RdbStorage::PutSchema(const Schema& schema) -> void {
  flatbuffers::FlatBufferBuilder builder;

  // Convert vector fields
  std::vector<flatbuffers::Offset<fb::VectorField>> vector_fields;
  for (const auto& field : schema.vector_fields) {
    auto fb_field =
        fb::CreateVectorField(builder, builder.CreateString(field.name),
                              field.dim, field.num_centroids);
    vector_fields.push_back(fb_field);
  }

  // Convert scalar fields
  std::vector<flatbuffers::Offset<fb::ScalarField>> scalar_fields;
  for (const auto& field : schema.scalar_fields) {
    fb::ScalarFieldType fb_type;
    switch (field.type) {
      case ScalarField::Type::kDouble:
        fb_type = fb::ScalarFieldType_kDouble;
        break;
      case ScalarField::Type::kInt:
        fb_type = fb::ScalarFieldType_kInt;
        break;
      case ScalarField::Type::kString:
        fb_type = fb::ScalarFieldType_kString;
        break;
      default:
        throw std::runtime_error("Unknown scalar field type");
    }

    auto fb_field = fb::CreateScalarField(
        builder, builder.CreateString(field.name), fb_type);
    scalar_fields.push_back(fb_field);
  }

  // Create schema
  auto fb_schema =
      fb::CreateSchema(builder, builder.CreateVector(vector_fields),
                       builder.CreateVector(scalar_fields));

  builder.Finish(fb_schema);

  // Store in RocksDB
  rocksdb::WriteOptions write_options;
  auto status = db_->Put(
      write_options, std::string(kSchemaPrefix),
      rocksdb::Slice(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize()));

  if (!status.ok()) {
    throw std::runtime_error("Failed to put schema: " + status.ToString());
  }
}

auto RdbStorage::GetSchema() const -> Schema {
  std::string value;
  rocksdb::ReadOptions read_options;
  auto status = db_->Get(read_options, std::string(kSchemaPrefix), &value);
  if (!status.ok()) {
    throw std::runtime_error("Failed to get schema: " + status.ToString());
  }

  const auto* const fb_schema = fb::GetSchema(value.data());
  Schema schema;

  for (const auto* fb_vector : *fb_schema->vector_fields()) {
    VectorField field;
    field.name = fb_vector->name()->str();
    field.dim = fb_vector->dim();
    field.num_centroids = fb_vector->num_centroids();
    schema.vector_fields.push_back(field);
  }

  for (const auto* fb_field : *fb_schema->scalar_fields()) {
    ScalarField::Type type;
    switch (fb_field->type()) {
      case fb::ScalarFieldType_kDouble:
        type = ScalarField::Type::kDouble;
        break;
      case fb::ScalarFieldType_kInt:
        type = ScalarField::Type::kInt;
        break;
      case fb::ScalarFieldType_kString:
        type = ScalarField::Type::kString;
        break;
      default:
        throw std::runtime_error("Unknown scalar field type in schema");
    }

    schema.AddScalarField(fb_field->name()->str(), type);
  }

  return schema;
}

auto RdbStorage::PutRecord(Key key, const Record& record) -> void {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<fb::Scalar>> fb_scalars;
  for (const auto& scalar : record.scalars) {
    flatbuffers::Offset<void> fb_value;
    fb::ScalarValue value_type;

    if (std::holds_alternative<double>(scalar)) {
      value_type = fb::ScalarValue_DoubleValue;
      fb_value =
          fb::CreateDoubleValue(builder, std::get<double>(scalar)).Union();
    } else if (std::holds_alternative<int>(scalar)) {
      value_type = fb::ScalarValue_IntValue;
      fb_value = fb::CreateIntValue(builder, std::get<int>(scalar)).Union();
    } else if (std::holds_alternative<std::string>(scalar)) {
      value_type = fb::ScalarValue_StringValue;
      fb_value =
          fb::CreateStringValue(
              builder, builder.CreateString(std::get<std::string>(scalar)))
              .Union();
    } else {
      throw std::runtime_error("Unknown scalar type");
    }
    auto fb_scalar = fb::CreateScalar(builder, value_type, fb_value);
    fb_scalars.push_back(fb_scalar);
  }

  std::vector<flatbuffers::Offset<fb::Vector>> fb_vectors;
  for (const auto& vector : record.vectors) {
    auto fb_vector = fb::CreateVector(
        builder, builder.CreateVector(vector.data(), vector.size()));
    fb_vectors.push_back(fb_vector);
  }

  auto fb_record =
      fb::CreateRecord(builder, record.id, builder.CreateVector(fb_scalars),
                       builder.CreateVector(fb_vectors));

  builder.Finish(fb_record);

  rocksdb::WriteOptions write_options;
  auto status = db_->Put(
      write_options, MakeRecordKey(key),
      rocksdb::Slice(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize()));
}

auto RdbStorage::GetRecord(Key key) const -> Record {
  std::string value;
  rocksdb::ReadOptions read_options;

  auto status = db_->Get(read_options, MakeRecordKey(key), &value);
  if (!status.ok()) {
    throw std::invalid_argument("Record not found");
  }

  Record result;
  const auto* const fb_record = flatbuffers::GetRoot<fb::Record>(value.data());
  result.id = fb_record->id();

  // Convert FlatBuffer scalars to C++ scalars
  const auto* fb_scalars = fb_record->scalars();
  if (fb_scalars) {
    for (const auto& fb_scalar : *fb_scalars) {
      switch (fb_scalar->value_type()) {
        case fb::ScalarValue_DoubleValue: {
          const auto* double_value = fb_scalar->value_as_DoubleValue();
          result.scalars.emplace_back(double_value->value());
          break;
        }
        case fb::ScalarValue_IntValue: {
          const auto* int_value = fb_scalar->value_as_IntValue();
          result.scalars.emplace_back(int_value->value());
          break;
        }
        case fb::ScalarValue_StringValue: {
          const auto* string_value = fb_scalar->value_as_StringValue();
          result.scalars.emplace_back(string_value->value()->str());
          break;
        }
        default:
          throw std::runtime_error("Unknown scalar type");
      }

    }  // for (const auto& fb_scalar : *fb_scalars)

    // Convert FlatBuffer vectors to C++ vectors
    const auto* fb_vectors = fb_record->vectors();
    if (fb_vectors) {
      for (const auto& fb_vector : *fb_vectors) {
        const auto* values = fb_vector->values();
        Vector vector;
        if (values) {
          vector.reserve(values->size());
          for (auto val : *values) {
            vector.push_back(val);
          }
        }
        result.vectors.push_back(std::move(vector));
      }
    }  // if (fb_vectors)
  }    // if (fb_scalars)
  return result;
}

auto RdbStorage::DeleteRecord(Key key) -> void {
  std::string record_key = MakeRecordKey(key);
  auto status = db_->Delete(rocksdb::WriteOptions(), record_key);
  if (!status.ok()) {
    throw std::runtime_error("Failed to delete record: " + status.ToString());
  }
}

auto RdbStorage::PutIndex(const std::string& field, const IvfFlatIndex& index)
    -> void {
  std::string index_key = MakeIndexKey(field);
  flatbuffers::FlatBufferBuilder builder;

  // Create field name
  auto field_name_offset = builder.CreateString(index.GetName());

  // Create centroids
  std::vector<flatbuffers::Offset<rox::fb::Vector>> centroids;
  for (const auto& centroid : index.GetCentroids()) {
    auto values = builder.CreateVector(centroid);
    auto vector_fb = rox::fb::CreateVector(builder, values);
    centroids.push_back(vector_fb);
  }

  // Create inverted lists
  std::vector<flatbuffers::Offset<rox::fb::IvfList>> inverted_lists;
  for (const auto& list : index.GetInvertedLists()) {
    std::vector<flatbuffers::Offset<rox::fb::IvfListEntry>> entries;

    for (const auto& [key, vec] : list) {
      auto values = builder.CreateVector(vec);
      auto vector_fb = rox::fb::CreateVector(builder, values);
      auto entry_fb = rox::fb::CreateIvfListEntry(builder, key, vector_fb);
      entries.push_back(entry_fb);
    }

    auto entries_vector = builder.CreateVector(entries);
    auto list_fb = rox::fb::CreateIvfList(builder, entries_vector);
    inverted_lists.push_back(list_fb);
  }

  // Create vectors
  auto centroids_vector = builder.CreateVector(centroids);
  auto lists_vector = builder.CreateVector(inverted_lists);

  // Create index
  auto fb_index =
      rox::fb::CreateIvfFlatIndex(builder, field_name_offset, index.dim_,
                                  index.nlist_, centroids_vector, lists_vector);

  // Finish the builder
  builder.Finish(fb_index);

  // Store in RocksDB
  rocksdb::WriteOptions write_options;
  auto status = db_->Put(
      write_options, index_key,
      rocksdb::Slice(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize()));
}

auto RdbStorage::GetIndex(const std::string& field)
    -> std::unique_ptr<IvfFlatIndex> {
  std::string index_key = MakeIndexKey(field);
  std::string value;
  rocksdb::ReadOptions read_options;
  auto status = db_->Get(read_options, index_key, &value);
  if (!status.ok()) {
    throw std::runtime_error("Failed to get index: " + status.ToString());
  }

  const auto* fb_index =
      flatbuffers::GetRoot<rox::fb::IvfFlatIndex>(value.data());
  // Create index
  std::string field_name = fb_index->field_name()->str();
  size_t dim = fb_index->dim();
  size_t nlist = fb_index->nlist();

  // IvfFlatIndex index(field_name, dim, nlist);
  std::unique_ptr<IvfFlatIndex> index =
      std::make_unique<IvfFlatIndex>(field_name, dim, nlist);

  // Extract centroids
  std::vector<Vector> centroids;
  if (fb_index->centroids()) {
    for (const auto* centroid : *fb_index->centroids()) {
      Vector vec;
      if (centroid->values()) {
        vec.reserve(centroid->values()->size());
        for (float val : *centroid->values()) {
          vec.push_back(val);
        }
      }
      centroids.push_back(std::move(vec));
    }
  }
  index->SetCentroids(centroids);

  // Extract inverted lists
  std::vector<IvfList> inverted_lists;
  if (fb_index->inverted_lists()) {
    for (const auto* list : *fb_index->inverted_lists()) {
      IvfList ivf_list;
      if (list->entries()) {
        for (const auto* entry : *list->entries()) {
          Key key = entry->key();
          Vector vec;
          if (entry->vector() && entry->vector()->values()) {
            vec.reserve(entry->vector()->values()->size());
            for (float val : *entry->vector()->values()) {
              vec.push_back(val);
            }
          }
          ivf_list.emplace_back(key, std::move(vec));
        }
      }
      inverted_lists.push_back(std::move(ivf_list));
    }
  }
  index->SetInvertedLists(inverted_lists);

  return index;
}

auto RdbStorage::DeleteIndex(const std::string& field) -> void {
  std::string index_key = MakeIndexKey(field);
  auto status = db_->Delete(rocksdb::WriteOptions(), index_key);
  if (!status.ok()) {
    throw std::runtime_error("Failed to delete index: " + status.ToString());
  }
}

}  // namespace rox