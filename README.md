<h1 align="center">TOP</h1>
<h3 align="center">
Smaller & <a href="https://github.com/petrizhang/TOP/edit/main/README.md">Faster</a> & Full-Fledged Similarity Search </br>
  Over <a href="https://github.com/petrizhang/TOP/edit/main/README.md">Hnswlib</a>
and <a href="https://github.com/petrizhang/TOP/edit/main/README.md">Faiss</a> Indexes (almost) for Free
</h3>
<br/>

---

- ✅ Up to 10x faster similarity search than HNSWLIB and FAISS
- ✅ Full-fledged top-k and range search support for HNSW and IVFPQ
- ✅ Transparent speedup without changing your index files
- ✅ Header-only library with easy-to-use python bindings


## Try It Now: 10x Search Performance with 3 Lines of Code

### Install
Ensure that there is a cxx17 compatible compiler installed in your system, and execute:
```bash
pip install pybind11
pip install topnn
```

### Boosted Approximate Nearest Neighbor Search over HNSW
```python
python -c """import topnn
searcher = topnn.create_fast_searcher("hnswlib", topnn.sample_hnsw_path, topnn.sample_pq_path)
print(searcher.ann_search([0.1, 0.2, 0.3, ... ], k=10))"""
```
