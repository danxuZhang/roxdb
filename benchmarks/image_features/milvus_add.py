#!/usr/bin/env python3

import h5py
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import argparse


def connect_to_milvus(host="localhost", port="19530"):
    """Connect to Milvus server"""
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f"Connected to Milvus server at {host}:{port}")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False


def create_collection(
    collection_name, vector_dim, index_type="IVF_FLAT", metric_type="L2"
):
    """Create a Milvus collection with appropriate schema"""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    # Convert numpy.int64 to regular Python int
    vector_dim = int(vector_dim)

    # Define fields for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="image_id", dtype=DataType.INT32),
        FieldSchema(name="category", dtype=DataType.INT32),
        FieldSchema(name="confidence", dtype=DataType.FLOAT),
        FieldSchema(name="votes", dtype=DataType.INT32),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]

    # Create collection schema
    schema = CollectionSchema(
        fields=fields, description=f"Vector collection with {vector_dim}D vectors"
    )

    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection: {collection_name} with dimension {vector_dim}")

    # Create an index on the vector field
    index_params = {
        "metric_type": metric_type,
        "index_type": index_type,
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"Created index on collection: {collection_name}")

    return collection


def import_h5_to_milvus(h5_file_path, host="localhost", port="19530"):
    """Import vector data from H5 file to Milvus collections"""
    # Connect to Milvus
    if not connect_to_milvus(host, port):
        return

    # Load H5 file
    try:
        with h5py.File(h5_file_path, "r") as f:
            # Get metadata and convert numpy types to Python native types
            num_records = int(f.attrs["num_records"])
            sift_dim = int(f.attrs["sift_dim"])
            gist_dim = int(f.attrs["gist_dim"])

            # Load vectors and metadata
            sift_vectors = f["sift"][:]
            gist_vectors = f["gist"][:]
            image_ids = f["image_id"][:]
            categories = f["category"][:]
            confidences = f["confidence"][:]
            votes = f["votes"][:]

            print(f"Loaded {num_records} records from {h5_file_path}")

            # Create collections for SIFT and GIST vectors
            sift_collection = create_collection("sift_vectors", sift_dim)
            gist_collection = create_collection("gist_vectors", gist_dim)

            # Prepare data for insertion
            batch_size = 10000
            for start_idx in range(0, num_records, batch_size):
                end_idx = min(start_idx + batch_size, num_records)
                batch_count = end_idx - start_idx

                # Prepare batch data - convert NumPy arrays to Python lists where needed
                ids = np.arange(start_idx, end_idx, dtype=np.int64).tolist()
                batch_image_ids = image_ids[start_idx:end_idx].tolist()
                batch_categories = categories[start_idx:end_idx].tolist()
                batch_confidences = confidences[start_idx:end_idx].tolist()
                batch_votes = votes[start_idx:end_idx].tolist()
                batch_sift = sift_vectors[start_idx:end_idx].tolist()
                batch_gist = gist_vectors[start_idx:end_idx].tolist()

                # Insert SIFT vectors
                sift_entities = [
                    ids,
                    batch_image_ids,
                    batch_categories,
                    batch_confidences,
                    batch_votes,
                    batch_sift,
                ]
                sift_collection.insert(sift_entities)

                # Insert GIST vectors
                gist_entities = [
                    ids,
                    batch_image_ids,
                    batch_categories,
                    batch_confidences,
                    batch_votes,
                    batch_gist,
                ]
                gist_collection.insert(gist_entities)

                print(
                    f"Inserted batch {start_idx+1}-{end_idx} of {num_records} records"
                )

            # Flush the collections to ensure all data is persisted
            sift_collection.flush()
            gist_collection.flush()

            # Load the collections into memory for better query performance
            sift_collection.load()
            gist_collection.load()

            # Get collection information
            print(f"SIFT collection entity count: {sift_collection.num_entities}")
            print(f"GIST collection entity count: {gist_collection.num_entities}")

            print("Import complete!")

    except Exception as e:
        print(f"Error importing data to Milvus: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Import H5 vector database to Milvus")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the H5 file containing vector data",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Milvus server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=str, default="19530", help="Milvus server port (default: 19530)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import_h5_to_milvus(args.input, args.host, args.port)


if __name__ == "__main__":
    main()
