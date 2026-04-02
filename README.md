<h1 align="center">TRIM: Accelerating Vector Search<br/>
via Enhanced Triangle-Inequality-Based Pruning</h1>

<p align="center">
Header-only C++ library with Python bindings for fast Top-K and range search on <b>HNSW</b> and <b>IVFPQ</b>
</p>

---

## Overview

**TRIM** (📄[Paper (SIGMOD 2026)](https://dl.acm.org/doi/pdf/10.1145/3769838)) is a general-purpose acceleration operator for high-dimensional vector similarity search (HVSS), based on enhanced triangle-inequality pruning.

It addresses a fundamental limitation in existing ANN systems:  
> **excessive data access and distance computations during search**

Unlike prior distance estimation or dimension-reduction approaches, TRIM:

- reduces **both data access and distance computation**
- maintains **high SIMD efficiency**
- seamlessly integrates into existing ANN indexes (e.g., HNSW, IVFPQ, and DiskANN)

---

## 🚀 Highlights

- **Up to 99% pruning ratio, 1.9× speedup over HNSWLIB and 3× speedup over FAISS (IVFPQ)**
- **Supports both top-k and range search**
- **Works with HNSW and IVFPQ**
- **Header-only library**
- **Python bindings for rapid experimentation**
- **SIMD-friendly pruning**
- **Easy integration into existing ANN pipelines**
- **Provides datasets for both top-k and range search evaluation**


---
> ```
> If you find these materials useful, please cite the following papers:
> ```

```
@article{trim,
  title={TRIM: Accelerating High-Dimensional Vector Similarity Search with Enhanced Triangle-Inequality-Based Pruning},
  author={Song, Yitong and Zhang, Pengcheng and Gao, Chao and Yao, Bin and Wang, Kai and Wu, Zongyuan and Qu, Lin},
  journal={Proceedings of the ACM on Management of Data},
  volume={3},
  number={6},
  pages={1--26},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```


## ⚙️ Installation

### 1. Environment Requirements

```bash
sudo apt install build-essential cmake libopenblas-dev liblapack-dev libhdf5-dev
pip install numpy pandas scipy h5py faiss-cpu hnswlib pybind11
```
---

### 2. Build & Install
```bash
# Build
git clone https://github.com/petrizhang/TRIM.git
cd TRIM
mkdir build
cd build
cmake ..
make -j

# Python Bindings
cd python
python setup.py build_ext --inplace
python setup.py install
```
---

### 3. Running Experiments
```bash
# Prepare directories
cd bench
mkdir -p ./tmp/index
mkdir -p ./results

# Download dataset from Hugging Face:
https://huggingface.co/datasets/songyitong/TRIM_Datasets

# Running (e.g., tHNSW and tIVFPQ)
# tHNSW
python3 bench.py -qt ann -k 10 -nq 1000 -d "/storage/vector_data/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k10.csv"

# tIVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "/storage/vector_data/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k10.csv"
```
--- 

### Acknowledgements 
We learned a lot from the following projects when building TRIM. 
- [faiss](https://github.com/facebookresearch/faiss)
- [hnswlib](https://github.com/nmslib/hnswlib)
- [glass](https://github.com/zilliztech/pyglass)

