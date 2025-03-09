#!/usr/bin/env python3

import h5py
import numpy as np
import argparse


def create_dataset(sift_vectors, gist_vectors, output_path, num_records):
    assert (
        sift_vectors.shape[1] == 128
    ), f"SIFT vectors should be 128D, got {sift_vectors.shape[1]}D"
    assert (
        gist_vectors.shape[1] == 960
    ), f"GIST vectors should be 960D, got {gist_vectors.shape[1]}D"

    if num_records > sift_vectors.shape[0] or num_records > gist_vectors.shape[0]:
        raise ValueError(
            f"Number of records {num_records} is greater than the number of SIFT vectors {sift_vectors.shape[0]}"
        )

    sift_vectors = sift_vectors[:num_records]
    gist_vectors = gist_vectors[:num_records]

    image_ids = np.arange(num_records, dtype=np.int32)
    categories = np.random.randint(0, 10, num_records).astype(np.int32)
    confidence = np.random.uniform(0, 1, num_records).astype(np.float32)
    votes = np.random.randint(0, 100, num_records).astype(np.int32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("sift", data=sift_vectors, compression="gzip")
        f.create_dataset("gist", data=gist_vectors, compression="gzip")

        f.create_dataset("image_id", data=image_ids)
        f.create_dataset("category", data=categories)
        f.create_dataset("confidence", data=confidence)
        f.create_dataset("votes", data=votes)

        f.attrs["num_records"] = num_records
        f.attrs["sift_dim"] = 128
        f.attrs["gist_dim"] = 960

    print(f"Dataset with {num_records} records successfully saved to {output_path}")


def read_fvecs(filename):
    with open(filename, "rb") as f:
        # Read dimension of feature vectors (first 4 bytes as int32)
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Move file pointer back to beginning
        f.seek(0)

        # Read all data
        data = np.fromfile(f, dtype=np.float32)

        # Reshape data - every d+1 values form a feature vector
        # (the first value is the dimension, then d actual values)
        # -1 means automatically determine the number of vectors
        vectors = data.reshape(-1, dim + 1)[:, 1:]

    return vectors


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare synthetic dataset")
    parser.add_argument(
        "--sift",
        type=str,
        required=True,
        help="Path to SIFT vectors in fvecs format",
    )
    parser.add_argument(
        "--gist",
        type=str,
        required=True,
        help="Path to GIST vectors in fvecs format",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the prepared dataset",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10000,
        help="Number of records in the dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sift_vectors = read_fvecs(args.sift)
    gist_vectors = read_fvecs(args.gist)
    create_dataset(sift_vectors, gist_vectors, args.output, args.n)


if __name__ == "__main__":
    main()
