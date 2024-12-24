# TOP
Faster ANN and Range Search for Hnswlib and Faiss Almost for Free

## 3x Search Performance with 3 Lines of Code

### Approximate Nearest Neighbor Search over HNSW
```python
import top
hnsw = top.hnsw.from_files("hnswlib_index_path", "faiss_index_pq_path")
hnsw.fast_ann_search([0.1, 0.2, 0.3, ... ], k=10, gamma=0.8)
```

### Range Search over HNSW
```python
import top
hnsw = top.hnsw.from_files("hnswlib_index_path", "faiss_index_pq_path")
hnsw.fast_range_search([0.1, 0.2, 0.3, ... ], radious=3, gamma=0.8)
```

### Approximate Nearest Neighbor Search over Faiss IVFPQ
```python
import top
ivfpq_rflat = top.ivfpq_rflat.from_files("faiss_index_ivf_pq_refine_flat_path")
ivfpq_rflat.fast_ann_search([0.1, 0.2, 0.3, ... ], k=10, gamma=0.8)
```

### Range Search over Faiss IVFPQ
```python
import top
ivfpq_rflat = top.ivfpq_rflat.from_files("faiss_index_ivf_pq_refine_flat_path")
ivfpq_rflat.fast_range_search([0.1, 0.2, 0.3, ... ], radious=3, gamma=0.8)
```
