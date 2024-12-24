# TOP
Faster ANN and Range Search for Hnswlib and Faiss Almost for Free

## 3x Search Performance with 3 Lines of Code

### Approximate Nearest Neighbor Search over HNSW
```python
import topnn
searcher = topnn.create_fast_searcher("hnswlib", "hnswlib_index_path", "faiss_index_pq_path")
D, I = searcher.ann_search([0.1, 0.2, 0.3, ... ], k=10)
```

### Range Search over HNSW
```python
import topnn
searcher = topnn.create_fast_searcher("hnswlib", "hnswlib_index_path", "faiss_index_pq_path")
D, I = searcher.range_search([0.1, 0.2, 0.3, ... ], radious=5)
```

### Approximate Nearest Neighbor Search over Faiss IVFPQ-RFLAT
```python
import topnn
searcher = topnn.create_fast_searcher("ivfpq_rflat", "faiss_index_ivf_pq_refine_flat_path")
D, I = searcher.ann_search([0.1, 0.2, 0.3, ... ], k=10)
```

### Range Search over Faiss IVFPQ-RFLAT
```python
import topnn
searcher = topnn.create_fast_searcher("ivfpq_rflat", "faiss_index_ivf_pq_refine_flat_path")
D, I = searcher.range_search([0.1, 0.2, 0.3, ... ], radious=5)
```
