<h1 align="center">TRIM</h1>
<h3 align="center">
Header-Only Library for Faster Top-K and Range Similarity Search</br>
  over <a href="https://github.com/nmslib/hnswlib">Hnswlib</a>
and <a href="https://github.com/facebookresearch/faiss">Faiss</a> Indexes
</h3>
<br/>

---

- ✅ 2x faster similarity search than HNSWLIB and FAISS
- ✅ Full-featured top-k and range search support for HNSW and IVFPQ
- ✅ Transparent speedup without changing your existing index files
- ✅ Header-only library with easy-to-use python bindings

## Try It Now

### Build
1. Ensure that there is a cxx17 compatible compiler (tested on gcc-10 and gcc-11) installed in your system.
2. Install requied Python dependencies:
  - python >= 3.8
  - faiss = 1.9.0
  - hnswlib = 0.8.0
  - pybind11 >= 2.11.1
3.  Build and install TRIM into your python environment:
```bash
cd python
python3 setup.py install
```

## Acknowledgements

We learned a lot from the following projects when building UNITY.
- [faiss](https://github.com/facebookresearch/faiss)
- [hnswlib](https://github.com/nmslib/hnswlib)
- [glass](https://github.com/zilliztech/pyglass)
