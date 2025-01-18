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
        self.p = hnswlib.Index(space=self.metric, dim=dim)
        self.searcher: unitylib.Searcher = None

    def fit(self, X):
        assert len(X[0]) == self.dim
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        assert self.searcher is not None
        self.searcher.set("ef", ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.searcher.ann_search(np.expand_dims(v, axis=0), k=n)

    def set_data(self, base_data):
        assert self.searcher is not None
        self.searcher.set_data(base_data)

    def save_index(self, index_path: str) -> None:
        self.p.save_index(index_path)
        self.load_index(index_path)

    def load_index(self, index_path: str) -> None:
        if self.searcher is None:
            creator = unitylib.SearcherCreator("hnsw")
            creator.set("hnswlib_index_path", index_path)
            creator.set("dim", self.dim)
            creator.set("metric", "L2")
            self.searcher = creator.create()

    def freeIndex(self):
        del self.p
