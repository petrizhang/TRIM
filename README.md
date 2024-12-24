<h1 align="center">TOP</h1>
<h3 align="center">
Smaller & <a href="https://github.com/petrizhang/TOP/edit/main/README.md">Faster</a> & Full-Fledged Similarity Search </br>
  Over <a href="https://github.com/petrizhang/TOP/edit/main/README.md">Hnswlib</a>
and <a href="https://github.com/petrizhang/TOP/edit/main/README.md">Faiss</a> Indexes (almost) for Free
</h3>
<br/>

---

- ✅ Up to 10x faster similarity search than HNSWLIB and FAISS
- ✅ Full-fledged top-k and range search support for HNSW and IVFPQ-RFLAT
- ✅ Transparent speedup without changing your index files
- ✅ Header-only library with easy-to-use python bindings


## Try It Now: 10x Search Performance with 3 Lines of Code

### Install

```bash
pip install topnn
```

### Approximate Nearest Neighbor Search over HNSW
```python
import topnn
searcher = topnn.create_fast_searcher("hnswlib", topnn.sample_hnsw_path, topnn.sample_pq_path)
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
