#!/usr/bin/env python3

import h5py
import numpy as np
import argparse
import time
from pymilvus import connections, Collection, utility


def connect_to_milvus(host="localhost", port="19530"):
    """Connect to Milvus server"""
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f"Connected to Milvus server at {host}:{port}")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False


def load_query_vectors(dataset_path):
    """Load the first SIFT and GIST vectors from dataset to use as query vectors"""
    try:
        with h5py.File(dataset_path, "r") as f:
            # Get metadata
            print(f"Loading query vectors from {dataset_path}")

            # Load first vectors for queries
            sift_query = f["sift"][0].tolist()
            gist_query = f["gist"][0].tolist()

            return sift_query, gist_query
    except Exception as e:
        print(f"Error loading query vectors: {e}")
        return None, None


def run_search_query(
    collection,
    vector_field,
    query_vector,
    filter_params=None,
    limit=100,
    search_params=None,
):
    """Execute a single search query on the collection"""
    if search_params is None:
        search_params = {"metric_type": "L2"}

    if filter_params:
        results = collection.search(
            data=[query_vector],
            anns_field=vector_field,
            param=search_params,
            limit=limit,
            expr=filter_params,
            output_fields=["image_id", "category", "confidence", "votes"],
        )
    else:
        results = collection.search(
            data=[query_vector],
            anns_field=vector_field,
            param=search_params,
            limit=limit,
            output_fields=["image_id", "category", "confidence", "votes"],
        )

    return results


def print_query_summary(
    query_name, avg_time, results=None, avg_scan_time=None, avg_recall=None
):
    """Print a summary of query performance"""
    print(f"\n===== {query_name} =====")
    print(f"Average search time: {avg_time:.2f} ms")

    if avg_scan_time is not None:
        print(f"Average scan time: {avg_scan_time:.2f} ms")

    if avg_recall is not None:
        print(f"Average recall: {avg_recall:.4f}")

    if results:
        print(f"Top 3 results:")
        for i, hit in enumerate(results[0][:3]):
            print(
                f"  #{i+1}: ID={hit.id}, Distance={hit.distance:.4f}, Category={hit.entity.get('category')}, "
                f"Confidence={hit.entity.get('confidence'):.4f}"
            )


def calculate_recall(search_results, ground_truth_results, k):
    """Calculate recall@k between search results and ground truth"""
    # Extract IDs from search results
    search_ids = set(hit.id for hit in search_results[0][:k])

    # Extract IDs from ground truth
    gt_ids = set(hit.id for hit in ground_truth_results[0][:k])

    # Calculate overlap
    intersection = search_ids.intersection(gt_ids)
    recall = len(intersection) / len(gt_ids) if gt_ids else 0

    return recall


def run_benchmark(
    dataset_path, evaluate=False, iterations=10, host="localhost", port="19530"
):
    """Run the benchmark queries against Milvus"""
    # Connect to Milvus
    if not connect_to_milvus(host, port):
        return

    # Check if collections exist
    if not utility.has_collection("sift_vectors") or not utility.has_collection(
        "gist_vectors"
    ):
        print("Error: sift_vectors or gist_vectors collection not found in Milvus.")
        print("Please import the vector database first using the import script.")
        return

    # Load collections
    sift_collection = Collection("sift_vectors")
    gist_collection = Collection("gist_vectors")

    # Load the collections into memory
    sift_collection.load()
    gist_collection.load()

    # Load query vectors
    sift_query, gist_query = load_query_vectors(dataset_path)
    if sift_query is None or gist_query is None:
        return

    # Define the search parameters
    search_params = {"metric_type": "L2", "params": {"nprobe": 32}}

    # Define the filter expression
    filter_expr = "category == 5 && confidence < 0.5"

    # Initialize the result storage
    queries = [
        {"name": "Q1: Single KNN on SIFT vectors", "times": []},
        {"name": "Q2: Single KNN on GIST vectors", "times": []},
        {"name": "Q3: Single KNN on SIFT vectors with filters", "times": []},
        {"name": "Q4: Single KNN on GIST vectors with filters", "times": []},
        {"name": "Q5: Multi KNN on SIFT and GIST vectors", "times": []},
        {"name": "Q6: Multi KNN on SIFT and GIST vectors with filters", "times": []},
    ]

    if evaluate:
        for q in queries:
            q["scan_times"] = []
            q["recalls"] = []

    # Run the benchmark for the specified number of iterations
    limit = 100  # k=100 as specified in the C++ code

    print(f"\nRunning benchmark with {iterations} iterations...")
    for i in range(iterations):
        print(f"\nIteration {i+1}")

        # Q1: Single KNN on SIFT vectors
        print("Query 1")
        start_time = time.time()
        q1_results = run_search_query(
            sift_collection,
            "vector",
            sift_query,
            limit=limit,
            search_params=search_params,
        )
        end_time = time.time()
        queries[0]["times"].append((end_time - start_time) * 1000)  # Convert to ms

        if evaluate:
            # Full scan for ground truth
            start_time = time.time()
            q1_ground_truth = run_search_query(
                sift_collection,
                "vector",
                sift_query,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            end_time = time.time()
            queries[0]["scan_times"].append((end_time - start_time) * 1000)
            queries[0]["recalls"].append(
                calculate_recall(q1_results, q1_ground_truth, limit)
            )

        # Q2: Single KNN on GIST vectors
        print("Query 2")
        start_time = time.time()
        q2_results = run_search_query(
            gist_collection,
            "vector",
            gist_query,
            limit=limit,
            search_params=search_params,
        )
        end_time = time.time()
        queries[1]["times"].append((end_time - start_time) * 1000)

        if evaluate:
            start_time = time.time()
            q2_ground_truth = run_search_query(
                gist_collection,
                "vector",
                gist_query,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            end_time = time.time()
            queries[1]["scan_times"].append((end_time - start_time) * 1000)
            queries[1]["recalls"].append(
                calculate_recall(q2_results, q2_ground_truth, limit)
            )

        # Q3: Single KNN on SIFT vectors with filters
        print("Query 3")
        start_time = time.time()
        q3_results = run_search_query(
            sift_collection,
            "vector",
            sift_query,
            filter_params=filter_expr,
            limit=limit,
            search_params=search_params,
        )
        end_time = time.time()
        queries[2]["times"].append((end_time - start_time) * 1000)

        if evaluate:
            start_time = time.time()
            q3_ground_truth = run_search_query(
                sift_collection,
                "vector",
                sift_query,
                filter_params=filter_expr,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            end_time = time.time()
            queries[2]["scan_times"].append((end_time - start_time) * 1000)
            queries[2]["recalls"].append(
                calculate_recall(q3_results, q3_ground_truth, limit)
            )

        # Q4: Single KNN on GIST vectors with filters
        print("Query 4")
        start_time = time.time()
        q4_results = run_search_query(
            gist_collection,
            "vector",
            gist_query,
            filter_params=filter_expr,
            limit=limit,
            search_params=search_params,
        )
        end_time = time.time()
        queries[3]["times"].append((end_time - start_time) * 1000)

        if evaluate:
            start_time = time.time()
            q4_ground_truth = run_search_query(
                gist_collection,
                "vector",
                gist_query,
                filter_params=filter_expr,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            end_time = time.time()
            queries[3]["scan_times"].append((end_time - start_time) * 1000)
            queries[3]["recalls"].append(
                calculate_recall(q4_results, q4_ground_truth, limit)
            )

        # Q5: Multi KNN on SIFT and GIST vectors (Milvus hybrid search)
        print("Query 5")
        start_time = time.time()
        # For hybrid search in Milvus, we need a different approach - we'll search both collections and combine results
        sift_results = run_search_query(
            sift_collection,
            "vector",
            sift_query,
            limit=limit,
            search_params=search_params,
        )
        gist_results = run_search_query(
            gist_collection,
            "vector",
            gist_query,
            limit=limit,
            search_params=search_params,
        )
        # In a real implementation, we would merge and rerank the results based on combined distance
        end_time = time.time()
        queries[4]["times"].append((end_time - start_time) * 1000)

        if evaluate:
            start_time = time.time()
            sift_ground_truth = run_search_query(
                sift_collection,
                "vector",
                sift_query,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            gist_ground_truth = run_search_query(
                gist_collection,
                "vector",
                gist_query,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            end_time = time.time()
            queries[4]["scan_times"].append((end_time - start_time) * 1000)
            # For simplicity, we'll average the recalls of both searches
            sift_recall = calculate_recall(sift_results, sift_ground_truth, limit)
            gist_recall = calculate_recall(gist_results, gist_ground_truth, limit)
            queries[4]["recalls"].append((sift_recall + gist_recall) / 2)

        # Q6: Multi KNN on SIFT and GIST vectors with filters
        print("Query 6")
        start_time = time.time()
        sift_results_filtered = run_search_query(
            sift_collection,
            "vector",
            sift_query,
            filter_params=filter_expr,
            limit=limit,
            search_params=search_params,
        )
        gist_results_filtered = run_search_query(
            gist_collection,
            "vector",
            gist_query,
            filter_params=filter_expr,
            limit=limit,
            search_params=search_params,
        )
        end_time = time.time()
        queries[5]["times"].append((end_time - start_time) * 1000)

        if evaluate:
            start_time = time.time()
            sift_ground_truth_filtered = run_search_query(
                sift_collection,
                "vector",
                sift_query,
                filter_params=filter_expr,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            gist_ground_truth_filtered = run_search_query(
                gist_collection,
                "vector",
                gist_query,
                filter_params=filter_expr,
                limit=limit,
                search_params={"metric_type": "L2", "params": {"nprobe": 1024}},
            )
            end_time = time.time()
            queries[5]["scan_times"].append((end_time - start_time) * 1000)
            sift_recall = calculate_recall(
                sift_results_filtered, sift_ground_truth_filtered, limit
            )
            gist_recall = calculate_recall(
                gist_results_filtered, gist_ground_truth_filtered, limit
            )
            queries[5]["recalls"].append((sift_recall + gist_recall) / 2)

    # Print the benchmark results
    print("\n===== BENCHMARK RESULTS =====")
    for i, query in enumerate(queries):
        avg_time = sum(query["times"]) / iterations

        if evaluate:
            avg_scan_time = sum(query["scan_times"]) / iterations
            avg_recall = sum(query["recalls"]) / iterations
            print_query_summary(
                query["name"],
                avg_time,
                results=eval(f"q{i+1}_results") if i < 4 else None,
                avg_scan_time=avg_scan_time,
                avg_recall=avg_recall,
            )
        else:
            print_query_summary(
                query["name"],
                avg_time,
                results=eval(f"q{i+1}_results") if i < 4 else None,
            )

    # Release the collections
    sift_collection.release()
    gist_collection.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Milvus search benchmark")
    parser.add_argument(
        "dataset_path", type=str, help="Path to the H5 file containing vector data"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate recall against full scan results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for each query (default: 10)",
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
    run_benchmark(
        args.dataset_path, args.evaluate, args.iterations, args.host, args.port
    )


if __name__ == "__main__":
    main()
