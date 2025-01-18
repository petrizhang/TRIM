<h1 align="center">UNITY</h1>
<h3 align="center">
Smaller & <a href="https://github.com/petrizhang/UNITY/edit/main/README.md">Faster</a> Full-Featured Similarity Search Engine</br>
  over <a href="https://github.com/petrizhang/UNITY/edit/main/README.md">Hnswlib</a>
and <a href="https://github.com/petrizhang/UNITY/edit/main/README.md">Faiss</a> Indexes
</h3>
<br/>

---

- ✅ Up to 10x faster similarity search than HNSWLIB and FAISS
- ✅ Full-featured top-k and range search support for HNSW and IVFPQ
- ✅ Transparent speedup without changing your existing index files
- ✅ Header-only library with easy-to-use python bindings

## Try It Now

### Install
Ensure that there is a cxx17 compatible compiler installed in your system, and execute:
```bash
pip install unitylib
```

### 10x Search Performance with 3 Lines of Code
```python
python -c """import unitylib
searcher = unitylib.create_fast_searcher("hnswlib", unitylib.sample_hnsw_path, unitylib.sample_pq_path)
print(searcher.ann_search([0.1, 0.2, 0.3, ... ], k=10))"""
```

## TODO
- [ ] Unified graph index format
- [ ] Seperated storage of graph index and data
- [ ] Support build indexes with UNITY directly
- [ ] Support PQ FastScan

## Acknowledgements

We learned a lot from the following projects when building UNITY.
- [faiss](https://github.com/facebookresearch/faiss)
- [hnswlib](https://github.com/nmslib/hnswlib)
- [glass](https://github.com/zilliztech/pyglass)
