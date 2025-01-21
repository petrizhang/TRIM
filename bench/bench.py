import argparse
import importlib
import itertools
import json
import os
import time
from typing import List

import pandas as pd
import utils
from alg import BaseANN
from utils import Timer


class DataSet:
    def __init__(self, base, query, groudtruth):
        self.base = base
        self.query = query
        self.groundtruth = groudtruth


def load_dataset(path: str):
    base, query, groundtruth = utils.read_hdf5_dataset(path,
                                                       ["train", "test", "neighbors"])
    return DataSet(base, query, groundtruth)


def enumerate_combinations(config_dict):
    keys = config_dict.keys()
    values = config_dict.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]


def parse_json_value(value_string):
    return json.loads(value_string)


def parse_index_config(config_string):
    config_dict = {}
    for group in config_string.split(";"):
        name, value_str = group.split(":")
        config_dict[name] = parse_json_value(value_str)
    return config_dict


def bench_epoch(alg: BaseANN, dataset: DataSet, k: int, nq: int, search_args: dict) -> List[dict]:
    line = dict(**search_args)
    alg.set_query_arguments(**search_args)
    if nq < 0 or nq > dataset.query.shape[0]:
        nq = dataset.query.shape[0]
    duration_ms = 0
    hit = 0
    total = 0
    print("="*40)
    print(f"Running queries under config {search_args}...")
    for i in range(nq):
        if i % 100 == 0:
            print(f"Running query {i}...")
        q = dataset.query[i]
        gt = dataset.groundtruth[i, :k]

        start = time.time()
        knn = alg.query(q, k)
        end = time.time()

        duration_ms += ((end-start) * 1000)
        gt_set = set(gt)
        total += len(gt_set)
        hit += len(gt_set & set(knn))
    recall = hit / total
    line["recall"] = recall
    line["latency(ms)"] = duration_ms / nq
    line["QPS"] = nq / (duration_ms / 1000)
    line["nq"] = nq
    return line


def bench(alg_class, dataset: DataSet, k: int, nq: int, build_args: dict,
          search_args: dict, save_index_path: str, save_result_path: str) -> None:
    dim = dataset.base.shape[1]
    # Build or load index
    if not os.path.exists(save_index_path):
        alg: BaseANN = alg_class(dim, build_args)
        print(f"Index not found at {
              save_index_path}. Creating and fitting a new index...")
        with Timer() as timer:
            alg.fit(dataset.base)
        duration = timer.elapsed_time
        print(f"Index built in {duration}s")
        if "empty" not in save_index_path:
            utils.write_build_time(save_index_path, duration)
        alg.save_index(save_index_path)

    print(f"Loading index from {save_index_path}...")
    alg = alg_class(dim, build_args)
    alg.load_index(save_index_path)

    alg.set_data(dataset.base)
    # Run queries, record time and recall
    search_args_combinations = enumerate_combinations(search_args)
    results = []
    for args in search_args_combinations:
        line = bench_epoch(alg, dataset, k, nq, args)
        results.append(line)
    results = pd.DataFrame(results)
    results.to_csv(save_result_path, index=None)
    print(f"Benchmark results for {alg.name} saved to {save_result_path}.")
    print(results)


class Args(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different indexing and search methods.")
    parser.add_argument("-k", "--k", required=True, type=int,
                        help="Number of neighbors to search")
    parser.add_argument("-d", "--dataset", required=True,
                        help="Path to the dataset.")
    parser.add_argument("-m", "--method", required=True,
                        choices=['hnsw', 'faiss', 'unity'], help="Method to test")
    parser.add_argument("-b", "--build_args", required=True,
                        help="Build parameters in the format: M:16;efConstruction:500")
    parser.add_argument("-s", "--search_args", required=True,
                        help="Search parameters in the format: ef:[10,20];use_bounded_queue:[true,false]")
    parser.add_argument("-nq", "--num_query", default=-1, required=False, type=int,
                        help="Number of queries to test")
    parser.add_argument("-si", "--save_index_path",
                        required=True, help="Path to save the index")
    parser.add_argument("-sr", "--save_result_path",
                        required=True, help="Path to save the results")
    args = parser.parse_args()

    # Only for debug
    # args = Args(dataset="./tmp/data/sift-128-euclidean.hdf5", method="top_hnsw",
    #             k=10,
    #             num_query=1000,
    #             build_args="M:16;efConstruction:500",
    #             search_args="use_bounded_queue:[false];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]",
    #             save_index_path="./tmp/index/sift_hnswlib16x500.bin",
    #             save_result_path="./tmp/results/sift_tophnsw16x500.csv")
    # args = Args(dataset="./tmp/data/sift-128-euclidean.hdf5", method="hnsw",
    #             k=10,
    #             num_query=1000,
    #             build_args="M:16;efConstruction:500",
    #             search_args="ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]",
    #             save_index_path="./tmp/index/sift_hnswlib16x500.bin",
    #             save_result_path="./tmp/results/sift_hnsw16x500.csv")

    # Parse build and search arguments
    build_args = parse_index_config(args.build_args)
    search_args = parse_index_config(args.search_args)

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Run benchmark
    alg_module = importlib.import_module(args.method)
    alg_class = getattr(alg_module, "Algorithm")
    bench(alg_class, dataset, args.k, args.num_query, build_args,
          search_args, args.save_index_path, args.save_result_path)


if __name__ == "__main__":
    main()
