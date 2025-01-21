import os

import faiss
import hnswlib
import numpy as np
import unitylib
from alg import BaseANN


class Algorithm(BaseANN):
    def __init__(self, dim, method_param):
        self.metric = "l2"
        self.method_param = method_param
        self.name = f"hnswlib ({self.method_param})"
        self.dim = dim
        self.hnsw = hnswlib.Index(space=self.metric, dim=dim)
        self.hnswlib_index_path = method_param["hnswlib_index_path"]
        self.dco = "exact"

        self.use_pq = False
        if "dco" in method_param and method_param["dco"] != "exact":
            self.dco = method_param["dco"]
            self.use_pq = True
            assert "pq_index_path" in method_param
            self.pq_index_path = method_param["pq_index_path"]
            self.pq_m = method_param["pq_m"]
            self.pq_nbits = method_param["pq_nbits"]
            # 8 is the length of the codes
            self.index_pq = faiss.IndexPQ(dim, self.pq_m, self.pq_nbits)

        self.searcher: unitylib.Searcher = None

    def fit(self, X):
        assert len(X[0]) == self.dim

        if not os.path.exists(self.hnswlib_index_path):
            print("Building hnswlib index...")
            # Only l2 is supported currently
            self.hnsw = hnswlib.Index(space=self.metric, dim=len(X[0]))
            self.hnsw.init_index(
                max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
            )
            data_labels = np.arange(len(X))
            self.hnsw.add_items(np.asarray(X), data_labels)
            self.hnsw.set_num_threads(1)
        if self.use_pq and not os.path.exists(self.pq_index_path):
            print("Building faiss PQ index...")
            self.index_pq.train(X)
            self.index_pq.add(X)

    def set_query_arguments(self, ef, enable_batch_dco=False, gamma=0.8):
        assert self.searcher is not None
        self.searcher.set("enable_batch_dco", enable_batch_dco)
        self.searcher.set("gamma", gamma)
        self.searcher.set("ef", ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.searcher.ann_search(np.expand_dims(v, axis=0), k=n)

    def set_data(self, base_data):
        assert self.searcher is not None
        self.searcher.set_data(base_data)

    def save_index(self, index_path: str) -> None:
        with open(index_path, "w") as f:
            pass
        if not os.path.exists(self.hnswlib_index_path):
            self.hnsw.save_index(self.hnswlib_index_path)
        if self.use_pq and not os.path.exists(self.pq_index_path):
            faiss.write_index(self.index_pq, self.pq_index_path)

    def load_index(self, index_path: str) -> None:
        if self.searcher is None:
            creator = unitylib.SearcherCreator("hnsw")
            creator.set("hnswlib_index_path", self.hnswlib_index_path)
            creator.set("dim", self.dim)
            creator.set("metric", "L2")
            creator.set("dco", self.dco)
            if self.use_pq:
                creator.set("pq_index_path", self.pq_index_path)
            self.searcher = creator.create()

    def freeIndex(self):
        del self.hnsw
