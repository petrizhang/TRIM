import os
import faiss
import numpy as np
import trimlib
import utils
from alg import BaseANN
from utils import Timer


class Algorithm(BaseANN):
    def __init__(self, dim, method_param: dict):
        print(f"method_param:{method_param}")
        self.metric = "l2"
        self.method_param = method_param
        self.name = f"tIVFPQfs ({self.method_param})"
        self.dim = dim
        self.ivfpq_fs = None
        self.ivfpq_index_path = method_param["ivfpqfs_index_path"]
        self.searcher: trimlib.Searcher = None

    def fit(self, X):
        assert len(X[0]) == self.dim

        if not os.path.exists(self.ivfpq_index_path):
            print("Building tIVFPQfs index...")
            nlist = self.method_param.get("C", 4096)
            self.m = self.method_param.get("m", 32)
            nbits = self.method_param.get("nbits", 4)

            quantizer = faiss.IndexPQFastScan(self.dim, self.m, nbits, faiss.METRIC_L2)
            self.ivfpq_fs = faiss.IndexIVFPQFastScan(quantizer, self.dim, nlist, self.m, nbits)
            self.ivfpq_fs.train(X)
            self.ivfpq_fs.add(X)


    def set_query_arguments(self, nprobe, gamma=0.8):
        assert self.searcher is not None
        self.searcher.set("nprobe", nprobe)
        self.searcher.set("gamma", gamma)
        self.searcher.clear_pruning_ratio()
        self.searcher.clear_num_distance_computation()

    def ann_query(self, v, n):
        return self.searcher.ann_search(np.expand_dims(v, axis=0), k=n)
    
    # def range_query(self, v, r):
    #     return self.searcher.range_search(np.expand_dims(v, axis=0), radius=r)
    
    def get_M(self):
        return self.m

    def get_pruning_ratio(self):
        return self.searcher.get_pruning_ratio()
    
    def get_actual_distance_computation(self):
        return self.searcher.get_actual_distance_computation()
    
    def get_total_distance_computation(self):
        return self.searcher.get_total_distance_computation()

    def set_data(self, base_data):
        assert self.searcher is not None
        self.searcher.set_data(base_data)

    def save_index(self, index_path: str) -> None:
        if not os.path.exists(index_path):
            faiss.write_index(self.ivfpq_fs, index_path)

    def load_index(self, index_path: str) -> None:
        if self.searcher is None:
            creator = trimlib.SearcherCreator("ivfpq_fs")
            creator.set("ivfpqfs_index_path", index_path)
            self.searcher = creator.create()

    def freeIndex(self):
        del self.ivfpq_fs
