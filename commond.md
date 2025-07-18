#### Scalability
### 2m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift2m.hdf5 -m trim -b "hnswlib_index_path:\"./tmp/index/sift2_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift2_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[100,200,300,400,500]" -si ./tmp/index/sift2_hnsw16x500_pq8x32.empty -sr ./results/Scalability_SIFT10M_tHNSW_KNN_k10.csv

python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift2m.hdf5 -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/sift2_ivfpq4096x8.bin";C:4096;m:8;nbits:8' -s 'trim_opened:[true];gamma:[0.5];k_factor:[300.0];nprobe:[200,300,350,400,450]' -si "./tmp/index/sift2_ivfpq4096x8.bin" -sr "./results/Scalability_SIFT10M_tIVFPQ_KNN_k10.csv"

### 4m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift4m.hdf5 -m trim -b "hnswlib_index_path:\"./tmp/index/sift4_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift4_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.8];ef:[30,50,100,200,300,400]" -si ./tmp/index/sift4_hnsw16x500_pq8x32.empty -sr ./results/Scalability_SIFT10M_t_KNN_k10.csv

python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift4m.hdf5 -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/sift4_ivfpq4096x8.bin";C:4096;m:8;nbits:8' -s 'trim_opened:[true];gamma:[0.48];k_factor:[300.0];nprobe:[300]' -si "./tmp/index/sift4_ivfpq4096x8.bin" -sr "./results/Scalability_SIFT10M_tIVFPQ_KNN_k10.csv"

### 6m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift6m.hdf5 -m trim -b "hnswlib_index_path:\"./tmp/index/sift6_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift6_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.8];ef:[1000]" -si ./tmp/index/sift6_hnsw16x500_pq8x32.empty -sr ./results/Scalability_SIFT10M_tHNSW_KNN_k10.csv

python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift6m.hdf5 -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/sift6_ivfpq4096x8.bin";C:4096;m:8;nbits:8' -s 'trim_opened:[true];gamma:[0.46];k_factor:[300.0];nprobe:[300]' -si "./tmp/index/sift6_ivfpq4096x8.bin" -sr "./results/Scalability_SIFT10M_tIVFPQ_KNN_k10.csv"

### 8m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift8m.hdf5 -m trim -b "hnswlib_index_path:\"./tmp/index/sift8_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift8_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.8];ef:[1000]" -si ./tmp/index/sift8_hnsw16x500_pq8x32.empty -sr ./results/Scalability_SIFT10M_tHNSW_KNN_k10.csv

python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift8m.hdf5 -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/sift8_ivfpq4096x8.bin";C:4096;m:8;nbits:8' -s 'trim_opened:[true];gamma:[0.45];k_factor:[300.0];nprobe:[300]' -si "./tmp/index/sift8_ivfpq4096x8.bin" -sr "./results/Scalability_SIFT10M_tIVFPQ_KNN_k10.csv"

### 10m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift10m.hdf5 -m trim -b "hnswlib_index_path:\"./tmp/index/sift10_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift10_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.8];ef:[1000]" -si ./tmp/index/sift10_hnsw16x500_pq8x32.empty -sr ./results/Scalability_SIFT10M_tHNSW_KNN_k10.csv

python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift10m.hdf5 -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/sift10_ivfpq4096x8.bin";C:4096;m:8;nbits:8' -s 'trim_opened:[true];gamma:[0.44];k_factor:[300.0];nprobe:[300]' -si "./tmp/index/sift10_ivfpq4096x8.bin" -sr "./results/Scalability_SIFT10M_tIVFPQ_KNN_k10.csv"

# parameter p and lambda
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67,0.88,0.89,0.9,0.95,0.97,0.99];ef:[1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/gist_thnsw16x500_pq8x120.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.79,0.89,0.92,0.95,0.97,0.99,1.0];ef:[1000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x32.empty" -sr "./results/nytimes_thnsw16x500_pq8x120.csv"

# parameter m
# GIST 
# tHNSW m=480
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x480.bin\";pq_m:480;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x480.empty" -sr "./results/parameterM_GIST.csv"
# tHNSW m=240
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x240.bin\";pq_m:240;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x240.empty" -sr "./results/parameterM_GIST.csv"
# tHNSW m=60
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x60.bin\";pq_m:60;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x60.empty" -sr "./results/parameterM_GIST.csv"
# tHNSW m=30
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x30.bin\";pq_m:30;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[300,400]" -si "./tmp/index/gist_thnsw16x500_pq8x30.empty" -sr "./results/parameterM_GIST.csv"
# tHNSW m=15
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x15.bin\";pq_m:15;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.6];ef:[300,400,500]" -si "./tmp/index/gist_thnsw16x500_pq8x15.empty" -sr "./results/parameterM_GIST.csv"
# tIVFPQ m=480
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x480.bin";C:4096;m:480;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[100,120,150]' -si "./tmp/index/gist_ivfpq4096x480.bin" -sr "./results/parameterM_GIST.csv"
# tIVFPQ m=240
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x240.bin";C:4096;m:240;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[100,120,150]' -si "./tmp/index/gist_ivfpq4096x240.bin" -sr "./results/parameterM_GIST.csv"
# tIVFPQ m=120
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[100,120,150]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/parameterM_GIST.csv"
# tIVFPQ m=60
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x60.bin";C:4096;m:60;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[100,120,150]' -si "./tmp/index/gist_ivfpq4096x60.bin" -sr "./results/parameterM_GIST.csv"
# tIVFPQ m=30
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[100,120,150]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/parameterM_GIST.csv"
# tIVFPQ m=15
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x15.bin";C:4096;m:15;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[100,120,150]' -si "./tmp/index/gist_ivfpq4096x15.bin" -sr "./results/parameterM_GIST.csv"


# Tiny5m
# tHNSW m=192
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x192.bin\";pq_m:192;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.63];ef:[200,300,400,500,600]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x192.empty" -sr "./results/parameterM_Tiny5M.csv"
# tHNSW m=48
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x48.bin\";pq_m:48;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[300,400,500,600]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x48.empty" -sr "./results/parameterM_Tiny5M.csv"
# tHNSW m=24
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x24.bin\";pq_m:24;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.6];ef:[300,400,500,600]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x24.empty" -sr "./results/parameterM_Tiny5M.csv"
# tHNSW m=12
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x12.bin\";pq_m:12;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.6];ef:[1500,2000,2500,3000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x12.empty" -sr "./results/parameterM_Tiny5M.csv"
# tHNSW m=6
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x6.bin\";pq_m:6;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.57];ef:[1500,2000,2500,3000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x6.empty" -sr "./results/parameterM_Tiny5M.csv"
# tIVFPQ m=192
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x192.bin";C:4096;m:192;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[100,150,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x192.bin" -sr "./results/parameterM_Tiny5M.csv"
# tIVFPQ m=96
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x96.bin";C:4096;m:96;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[100,150,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x96.bin" -sr "./results/parameterM_Tiny5M.csv"
# tIVFPQ m=48
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x48.bin";C:4096;m:48;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[100,150,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x48.bin" -sr "./results/parameterM_Tiny5M.csv"
# tIVFPQ m=24
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x24.bin";C:4096;m:24;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[100,150,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x24.bin" -sr "./results/parameterM_Tiny5M.csv"
# tIVFPQ m=12
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x12.bin";C:4096;m:12;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[100,150,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x12.bin" -sr "./results/parameterM_Tiny5M.csv"
# tIVFPQ m=6
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x6.bin";C:4096;m:6;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[100,150,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x6.bin" -sr "./results/parameterM_Tiny5M.csv"


# NYTimes
# tHNSW m=128
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x128.bin\";pq_m:128;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[10,50,90,150,300,400,600,800,1000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x128.empty" -sr "./results/parameterM_NYTimes.csv"
# tHNSW m=64
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[100,150,200,250,300,350]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/parameterM_NYTimes.csv"
# tHNSW m=32
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[10,50,90,150,300,400,600,800,1000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x32.empty" -sr "./results/parameterM_NYTimes.csv"
# tHNSW m=16
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x16.bin\";pq_m:16;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.78,0.77];ef:[1000,1200,1500]" -si "./tmp/index/nytimes_thnsw16x500_pq8x16.empty" -sr "./results/parameterM_NYTimes.csv"
# tHNSW m=8
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x8.bin\";pq_m:8;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[400,600,800,1000,2000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x8.empty" -sr "./results/parameterM_NYTimes.csv"
# tHNSW m=4
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x4.bin\";pq_m:4;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[10,50,90,150,300,400,600,800,1000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x4.empty" -sr "./results/parameterM_NYTimes.csv"
# tIVFPQ m=128
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x128.bin";C:4096;m:128;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[1200,1500,1800,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x128.bin" -sr "./results/parameterM_NYTimes.csv"
# tIVFPQ m=64
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x64.bin";C:4096;m:64;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[1200,1500,1800,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x64.bin" -sr "./results/parameterM_NYTimes.csv"
# tIVFPQ m=32
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x32.bin";C:4096;m:32;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[1200,1500,1800,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x32.bin" -sr "./results/parameterM_NYTimes.csv"
# tIVFPQ m=16
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[1200,1500,1800,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/parameterM_NYTimes.csv"
# tIVFPQ m=8
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x8.bin";C:4096;m:8;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[1200,1500,1800,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x8.bin" -sr "./results/parameterM_NYTimes.csv"
# tIVFPQ m=4
python3 parameterM.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[1200,1500,1800,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x4.bin" -sr "./results/parameterM_NYTimes.csv"


##### HNSW-ann
# GIST
# tHNSW
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k10.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[60,100,150,200,300,400,500,1100,1700]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k10.csv"
# tHNSW (random landmarks)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;random_landmark_size:10;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.02];ef:[60,100,150,200,300,400,500,1100,1700]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k10.csv"
# tHNSW
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[100,200,300,400,600,800,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k100.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[200,400,600,800,1000,1500,2000,3000,4000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k100.csv"
# tHNSW (random landmarks)
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;random_landmark_size:10;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.02];ef:[200,400,600,800,1000,1500,2000,3000,4000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k100.csv"

# GloVe
# tHNSW
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.61];ef:[10,30,70,150,400,800,1000,1500,3000,4000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k10.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[100,200,400,600,800,1000,2000,3500,4500]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k10.csv"
# tHNSW (random landmarks)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;random_landmark_size:10;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.2];ef:[100,200,400,600,800,1000,2000,3500,4500]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k10.csv"
# tHNSW
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[100,160,200,400,800,1000,1200,1500,2000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k100.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[200,400,600,800,1000,2000,4000,6000,8000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k100.csv"
# tHNSW (random landmarks)
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;random_landmark_size:10;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.2];ef:[200,400,600,800,1000,2000,4000,6000,8000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k100.csv"

# NYTimes
# tHNSW
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[10,50,90,150,300,400,600,800,1000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_KNN_k10.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[60,100,200,300,500,1000,2000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_KNN_k10.csv"
# tHNSW (random landmarks)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;random_landmark_size:10;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.2];ef:[60,100,200,300,500,1000,2000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_KNN_k10.csv"

# Tiny5m
# tHNSW
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[10,50,90,150,300,600,800,1000,2000,3000,4000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_KNN_k10.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[60,100,200,300,500,1000,1500,2000,2500,3000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_KNN_k10.csv"
# tHNSW (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;random_landmark_size:10;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.006];ef:[60,100,200,300,500,1000,1500,2000,2500,3000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_KNN_k10.csv"


#### HNSW-range
# GIST
# tHNSW
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_ARS_0.01%.csv"
# tHNSW (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[200,300,400,500,700,900,1100,1300,1500]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_ARS_0.01%.csv"
# tHNSW
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.69];ef:[350,400,450,500,550,600,650,700,800,900,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_ARS_0.1%.csv"
# tHNSW (without pLB)
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[800,900,1000,1100,1200,1300,1500,1700,1900,2400,3000,4000,5000,6000,8000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_ARS_0.1%.csv"

# GloVe
# tHNSW
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.63];ef:[70,90,200,400,600,800,1000,1200,1500,2000,2500,3000,3500,4000,5000,6000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_ARS_0.01%.csv"
# tHNSW (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[300,500,700,1000,1300,1600,1900,2100,2400,3000,3500,4000,4500,5000,6000,8000,10000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_ARS_0.01%.csv"
# tHNSW
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[450,500,550,600,650,700,750,800,850,900,950,1000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_ARS_0.1%.csv"
# tHNSW (without pLB)
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_ARS_0.1%.csv"

# NYTimes
# tHNSW
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.75];ef:[30,50,100,150,300,400,600,800,1000,3000,5000,7000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_ARS_0.01%.csv"
# tHNSW (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[100,200,300,500,700,900,1100,1300,1500,2000,3000,4000,5000,6000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_ARS_0.01%.csv"

# Tiny5m
# tHNSW
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[100,120,150,200,250,300,400,600,800,1000,2000,3000,4000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_ARS_0.01%.csv"
# tHNSW (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.0];ef:[500,800,1200,1500,2000,2500,3000,4000,5000,6000,7000,8000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_ARS_0.01%.csv"


#### IVFPQ-knn
# GIST m=30
# IVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[600.0];nprobe:[20,40,60,80,100,150,200,250,300,400]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_KNN_k10.csv"
# tIVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k10.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[0.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k10.csv"
# IVFPQ
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x32.bin";C:4096;m:32;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[600.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x32.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_KNN_k100.csv"
# tIVFPQ
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x32.bin";C:4096;m:32;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[40,60,80,100,120,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x32.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k100.csv" 
# tIVFPQ (without pLB)
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x32.bin";C:4096;m:32;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[0.0];nprobe:[40,60,80,100,120,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x32.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k100.csv"

# GloVe m=4
# IVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[6000.0];nprobe:[40,60,80,100,200,300,400,600,800,1000,1200]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_KNN_k10.csv"
# tIVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[0.0];nprobe:[40,60,80,100,200,300,400,600,800,1000,1200,1400,1600,1800,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_KNN_k10.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[0.0];nprobe:[40,60,80,100,200,300,400,600,800,1000,1200,1400,1600,1800,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_KNN_k10.csv"
# IVFPQ
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[6000.0];nprobe:[40,60,80,100,200,300,400,500,600,700,800,900,1000,1100,1200]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_KNN_k100.csv"
# tIVFPQ
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[0.0];nprobe:[40,60,80,100,200,400,600,800,1000,1200,1400,1600]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_KNN_k100.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[0.0];nprobe:[40,60,80,100,200,400,600,800,1000,1200,1400,1600]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_KNN_k100.csv"

# NYTimes m=16
# IVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[8000.0];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_IVFPQ_KNN_k10.csv"
# tIVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000,4000]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQ_KNN_k10.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[0.0];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000,4000]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQ_KNN_k10.csv"

# Tiny5m m=16
# IVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[4000.0];nprobe:[5,8,10,15,20,30,40,50,60,70,80,90,100,200,300]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_IVFPQ_KNN_k10.csv"
# tIVFPQ
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[5,8,10,15,20,30,40,50,60,70,80,90,100,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQ_KNN_k10.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[0.0];nprobe:[5,8,10,15,20,30,40,50,60,70,80,90,100,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQ_KNN_k10.csv"


#### IVFPQ-range
# GIST m=30
# IVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.8];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_ARS_0.01%.csv"
# tIVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[1.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_ARS_0.01%.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[1.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_ARS_0.01%.csv"
# IVFPQ
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.8];nprobe:[500,600,700,800,900,1000,1100]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_ARS_0.1%.csv"
# tIVFPQ
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[1.0];nprobe:[100,150,200,250,300,400,500,600,700,800,900,1000]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_ARS_0.1%.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[1.0];nprobe:[100,150,200,250,300,400,500,600,700,800,900,1000]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_ARS_0.1%.csv"

# GloVe m=4
# IVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.3];nprobe:[700,750,800,900,1000,1100,1200,1300,1400]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_ARS_0.01%.csv"
# tIVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[1.0];nprobe:[700,750,800,900,1000,1100,1200,1300,1400,1500]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_ARS_0.01%.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[1.0];nprobe:[700,750,800,900,1000,1100,1200,1300,1400,1500]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_ARS_0.01%.csv"
# IVFPQ
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.3];nprobe:[1200,1300,1400,1500,1600,1700,1800,1900,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_ARS_0.1%.csv"
# tIVFPQ
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[1.0];nprobe:[1200,1300,1400,1500,1600,1700,1800,1900,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_ARS_0.1%.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[1.0];nprobe:[1200,1300,1400,1500,1600,1700,1800,1900,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_ARS_0.1%.csv"

# NYTimes m=16
# IVFPQ 
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.3];nprobe:[800,900,1000,1200,1500,1800,2000,2500,3000,3500]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_IVFPQ_ARS_0.01%.csv"
# tIVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.67];k_factor:[1.0];nprobe:[800,900,1000,1200,1500,1800,2000,2500,3000,3500]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQ_ARS_0.01%.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[1.0];nprobe:[800,900,1000,1200,1500,1800,2000,2500,3000,3500]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQ_ARS_0.01%.csv"

# Tiny5m m=16
# IVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.0];nprobe:[30,50,80,100,150,200,250,300,350,400,450,500]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_IVFPQ_ARS_0.01%.csv"
# tIVFPQ
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.57];k_factor:[1.0];nprobe:[100,150,200,250,300,350,400,450,500]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQ_ARS_0.01%.csv"
# tIVFPQ (without pLB)
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.0];k_factor:[1.0];nprobe:[100,150,200,250,300,350,400,450,500]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQ_ARS_0.01%.csv"


# FastScan
# IVFPQfs
# GIST m=480 
python3 bench.py -qt ann -k 10 -nq 100 -d "../../yitong/Datasets/gist-960.hdf5" -m IVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/gist_ivfpqfs4096x480.bin";C:4096;m:480;nbits:4' -s 'k_factor:[50];nprobe:[10,20,30,40,50,60,70,80,100,150,200,250,300,400,500]' -si "./tmp/index/gist_ivfpqfs4096x480.bin" -sr "./results/QPSDCRecall_GIST_IVFPQfs_KNN_k10.csv"

# Glove m=50 (取了k_factor=100的)
python3 bench.py -qt ann -k 10 -nq 100 -d "../../yitong/Datasets/glove-100.hdf5" -m IVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/glove_ivfpqfs4096x50.bin";C:4096;m:50;nbits:4' -s 'k_factor:[80,100];nprobe:[100,200,400,600,800,1000,1200,1400,1600]' -si "./tmp/index/glove_ivfpqfs4096x50.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQfs_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 100 -d "../../yitong/Datasets/glove-100.hdf5" -m IVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/glove_ivfpqfs4096x50.bin";C:4096;m:50;nbits:4' -s 'k_factor:[3,5,10,20,30,40,50];nprobe:[60,100,200,400,600,800,1000,1200,1400,1600]' -si "./tmp/index/glove_ivfpqfs4096x50.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQfs_KNN_k10.csv"

# NYTimes m=128
python3 bench.py -qt ann -k 10 -nq 100 -d "../../yitong/Datasets/nytimes-256_normalized.hdf5" -m IVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/nytimes_ivfpqfs4096x128.bin";C:4096;m:128;nbits:4' -s 'k_factor:[60,70,80];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000]' -si "./tmp/index/nytimes_ivfpqfs4096x128.bin" -sr "./results/QPSDCRecall_NYTimes_IVFPQfs_KNN_k10.csv"

# Tiny5M m=192
python3 bench.py -qt ann -k 10 -nq 500 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m IVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/tiny5m_ivfpqfs4096x192.bin";C:4096;m:192;nbits:4' -s 'k_factor:[60,70,80];nprobe:[20,30,40,50,60,70,80,90,100,200,300,400,500]' -si "./tmp/index/tiny5m_ivfpqfs4096x192.bin" -sr "./results/QPSDCRecall_Tiny5m_IVFPQfs_KNN_k10.csv"


# tIVFPQfs
# GIST m=480 
python3 bench.py -qt ann -k 10 -nq 500 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/gist_tivfpqfs4096x480.bin";C:4096;m:480;nbits:4' -s 'gamma:[0.6,0.65,0.68,0.7,0.72,0.75,0.8];nprobe:[40,50,60,70,80,90,100,200,300,500]' -si "./tmp/index/gist_tivfpqfs4096x480.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQfs_KNN_k10.csv"

# Glove m=50
python3 bench.py -qt ann -k 10 -nq 500 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/glove_tivfpqfs4096x50.bin";C:4096;m:50;nbits:4' -s 'gamma:[0.5,0.55,0.6,0.65,0.7,0.75,0.8];nprobe:[60,100,200,400,600,800,1000,1200,1400,1600]' -si "./tmp/index/glove_tivfpqfs4096x50.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQfs_KNN_k10.csv"

# NYTimes m=128
python3 bench.py -qt ann -k 10 -nq 500 -d "../../yitong/Datasets/nytimes-256_normalized.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/nytimes_tivfpqfs4096x128.bin";C:4096;m:128;nbits:4' -s 'gamma:[0.5,0.55,0.6,0.65,0.7,0.75,0.8];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000]' -si "./tmp/index/nytimes_tivfpqfs4096x128.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQfs_KNN_k10.csv"

# Tiny5M m=192
python3 bench.py -qt ann -k 10 -nq 500 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m IVFPQfs -b 'ivfpqfs_index_path:"./tmp/index/tiny5m_tivfpqfs4096x192.bin";C:4096;m:192;nbits:4' -s 'gamma:[0.5,0.55,0.6,0.65,0.7,0.75,0.8];nprobe:[20,30,40,50,60,70,80,90,100,200,300,400,500]' -si "./tmp/index/tiny5m_tivfpqfs4096x192.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQfs_KNN_k10.csv"
