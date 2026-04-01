import os

import faiss
import numpy as np
import utils
from alg import BaseANN
from utils import Timer


class Algorithm(BaseANN):
    def __init__(self, dim, method_param: dict):
        print("Faiss version:", faiss.__version__)
        print(f"method_param:{method_param}")
        self.metric = "METRIC_L2"
        self.method_param = method_param
        self.name = f"IVFPQfs ({self.method_param})"
        self.dim = dim
        self.ivfpq_fs = None
        self.ivfpq_index_path = method_param["ivfpqfs_index_path"]
        self.nq = 0

    def fit(self, X):
        assert len(X[0]) == self.dim
        if not os.path.exists(self.ivfpq_index_path):
            print("Building IVFPQfs index...")
            nlist = self.method_param.get("C", 4096)
            self.m = self.method_param.get("m", self.dim // 2)
            nbits = self.method_param.get("nbits", 4)
            
            index_string = f"IVF{nlist},PQ{self.m}x{nbits}fs,RFlat"
            self.ivfpq_fs = faiss.index_factory(self.dim, index_string, faiss.METRIC_L2)
            self.ivfpq_fs.train(X)
            self.ivfpq_fs.add(X)
                

    def set_query_arguments(self, nprobe, k_factor):
        faiss.omp_set_num_threads(1)
        self.base_index = faiss.downcast_index(self.ivfpq_fs.base_index)
        self.base_index.nprobe = nprobe
        self.ivfpq_fs.k_factor = k_factor
        self.k_factor = k_factor



    def ann_query(self, v, n):
        self.nq += 1
        self.k = n

        D, I = self.ivfpq_fs.search(np.expand_dims(v,axis=0), n)
        return I[0]

       
    def get_pruning_ratio(self):
        return 0
    
    def get_actual_distance_computation(self):
        self.dc = self.k * self.k_factor * self.nq
        self.nq = 0
        return self.dc
    
    def get_total_distance_computation(self):
        return self.dc

    def set_data(self, base_data):
        self.X = base_data

    def save_index(self, index_path: str) -> None:
        if not os.path.exists(index_path):
            faiss.write_index(self.ivfpq_fs, index_path)

    def load_index(self, index_path: str) -> None:
        self.ivfpq_fs = faiss.read_index(index_path)

    def freeIndex(self):
        del self.ivfpq_fs

