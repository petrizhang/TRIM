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
        self.name = f"tIVFPQ ({self.method_param})"
        self.dim = dim
        self.ivfpq = None
        self.ivfpq_index_path = method_param["ivfpq_index_path"]
        self.searcher: trimlib.Searcher = None

    def fit(self, X):
        assert len(X[0]) == self.dim

        if not os.path.exists(self.ivfpq_index_path):
            print("Building IVFPQ index...")
            # with Timer() as timer:
            quantizer = faiss.IndexFlatL2(self.dim)
            nlist = self.method_param.get("C", 1024)
            m = self.method_param.get("m", 32)
            nbits = self.method_param.get("nbits", 8)

            self.ivfpq = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits)
            self.ivfpq.train(X)
            self.ivfpq.add(X)

                # faiss.write_index(self.ivfpq, self.ivfpq_index_path)

            # utils.write_build_time(self.ivfpq_index_path, timer.elapsed_time)


    def set_query_arguments(self, nprobe, k_factor, trim_opened = True, gamma=0.8):
        assert self.searcher is not None
        self.searcher.set("nprobe", nprobe)
        self.searcher.set("gamma", gamma)
        # self.searcher.set("ef", ef)
        self.searcher.set("k_factor", k_factor)
        self.searcher.set("trim_opened", trim_opened)
        self.searcher.clear_pruning_ratio()

    def ann_query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.searcher.ann_search(np.expand_dims(v, axis=0), k=n)
    
    def range_query(self, v, r):
        return self.searcher.range_search(np.expand_dims(v, axis=0), radius=r)
    
    def get_pruning_ratio(self):
        return self.searcher.get_pruning_ratio()
    

    def set_data(self, base_data):
        assert self.searcher is not None
        self.searcher.set_data(base_data)

    def save_index(self, index_path: str) -> None:
        if not os.path.exists(self.ivfpq_index_path):
            faiss.write_index(self.ivfpq, self.ivfpq_index_path)

    def load_index(self, index_path: str) -> None:
        if self.searcher is None:
            creator = trimlib.SearcherCreator("ivfpq")
            creator.set("ivfpq_index_path", self.ivfpq_index_path)
            self.searcher = creator.create()

    def freeIndex(self):
        del self.ivfpq
