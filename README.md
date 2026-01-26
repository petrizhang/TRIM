<h1 align="center">TRIM</h1>
<h3 align="center">
Header-Only Library for Faster Top-K and Range Similarity Search</br>
  over <b>Hnswlib</b>
and <b>Faiss</b> Indexes
</h3>
<br/>

---

- ✅ Up to 3x faster similarity search than HNSWLIB and FAISS
- ✅ Full-featured top-k and range search support for HNSW and IVFPQ
- ✅ Header-only library with easy-to-use python bindings

## Try It Now

### 1. Environment Requirements 
```bash
sudo apt install build-essential cmake libopenblas-dev liblapack-dev libhdf5-dev
pip install numpy pandas scipy h5py faiss-cpu hnswlib pybind11
```
---

## 2. Build & Install
### build
```bash
git clone https://github.com/petrizhang/TRIM.git
cd TRIM
mkdir build
cd build
cmake ..
make -j
```
---
### install
```bash
cd python
python setup.py build_ext --inplace
python setup.py install
```
---

### 3. Running Experiments

```bash
cd bench
mkdir -p ./tmp/index
mkdir -p ./results
```
#### tHNSW
```bash
# eg. GIST
python3 bench.py -qt ann -k 10 -nq 1000 -d "/storage/vector_data/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k10.csv"
```
#### tIVFPQ
```bash
# eg.GIST
python3 bench.py -qt ann -k 10 -nq 1000 -d "/storage/vector_data/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k10.csv"
```
---

### Acknowledgements

We learned a lot from the following projects when building TRIM.
- [faiss](https://github.com/facebookresearch/faiss)
- [hnswlib](https://github.com/nmslib/hnswlib)
- [glass](https://github.com/zilliztech/pyglass)
