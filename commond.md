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

##### HNSW-ann
# GIST
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k10.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[3500,4000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_KNN_k100.csv"
<!-- 100,200,300,400,600,800,1000,1200,1400,1600,1800,2000,2500,3000 -->

# GloVe
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.61];ef:[10,30,50,70,90,150,300,400,600,800,1000,1500,2000,3000,4000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k10.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[100,200,400,600,800,1000,1200,1500]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_KNN_k100.csv"

# NYTimes
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[10,50,90,150,300,400,600,800,1000,3000,5000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_KNN_k10.csv"

# Tiny5m
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[10,50,90,150,300,600,800,1000,2000,3000,4000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_KNN_k10.csv"


#### HNSW-range
# GIST
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[350,400,450,500,550,600,650,700,800,900,1000]" -si "./tmp/index/gist_thnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_tHNSW_ARS_0.1%.csv"

# GloVe
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.63];ef:[5500,6000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_ARS_0.01%.csv"
<!-- 50,70,90,100,200,400,600,800,1000,1200,1500,2000,2500,3000,3500,4000,5000 -->

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[900,950,1000]" -si "./tmp/index/glove_thnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_tHNSW_ARS_0.1%.csv"

# NYTimes
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.75];ef:[30,50,100,150,300,400,600,800,1000,3000,5000,7000]" -si "./tmp/index/nytimes_thnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_tHNSW_ARS_0.01%.csv"

# Tiny5m
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m trim -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"trim\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[100,120,150,200,250,300,400,600,800,1000,2000,3000,4000]" -si "./tmp/index/tiny5m_thnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_tHNSW_ARS_0.01%.csv"


#### IVFPQ-knn
# GIST m=30
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[600.0];nprobe:[20,40,60,80,100,150,200,250,300,400]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x32.bin";C:4096;m:32;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[600.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x32.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_KNN_k100.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x32.bin";C:4096;m:32;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[0.0];nprobe:[40,60,80,100,120,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x32.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_KNN_k100.csv"

# GloVe m=4
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[6000.0];nprobe:[40,60,80,100,200,300,400,600,800,1000,1200]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.6];k_factor:[0.0];nprobe:[40,60,80,100,200,400,600,800,1000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[6000.0];nprobe:[40,60,80,100,200,300,400,500,600,700,800,900,1000,1100,1200]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_KNN_k100.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[0.0];nprobe:[40,60,80,100,200,400,600,800,1000,1200,1400,1600]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_KNN_k100.csv"

# NYTimes m=16
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[8000.0];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_IVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[0.0];nprobe:[20,40,60,80,100,150,200,400,600,800,1000,2000,3000,4000]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQ_KNN_k10.csv"

# Tiny5m m=16
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[4000.0];nprobe:[5,8,10,15,20,30,40,50,60,70,80,90,100,200,300]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_IVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.53];k_factor:[0.0];nprobe:[5,8,10,15,20,30,40,50,60,70,80,90,100,200,300,400]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQ_KNN_k10.csv"


#### IVFPQ-range
# GIST m=30
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.8];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[1.0];nprobe:[40,60,80,100,150,200,250,300,400,500,600,700,800]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.9];nprobe:[500,600,700,800,900,1000,1100]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_ARS_0.1%.csv"

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x30.bin";C:4096;m:30;nbits:8' -s 'trim_opened:[true];gamma:[0.62];k_factor:[1.0];nprobe:[100,150,200,250,300,400,500,600,700,800,900,1000]' -si "./tmp/index/gist_ivfpq4096x30.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQ_ARS_0.1%.csv"

# GloVe m=4
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.3];nprobe:[700,750,800,900,1000,1100,1200,1300,1400]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[1.0];nprobe:[700,750,800,900,1000,1100,1200,1300,1400,1500]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.3];nprobe:[1200,1300,1400,1500,1600,1700,1800,1900,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_IVFPQ_ARS_0.1%.csv"

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/glove_ivfpq4096x4.bin";C:4096;m:4;nbits:8' -s 'trim_opened:[true];gamma:[0.55];k_factor:[1.0];nprobe:[1200,1300,1400,1500,1600,1700,1800,1900,2000]' -si "./tmp/index/glove_ivfpq4096x4.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQ_ARS_0.1%.csv"

# NYTimes m=16
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.3];nprobe:[800,900,1000,1200,1500,1800,2000,2500,3000,3500]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_IVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/nytimes_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.63];k_factor:[1.0];nprobe:[800,900,1000,1200,1500,1800,2000,2500,3000,3500]' -si "./tmp/index/nytimes_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_NYTimes_tIVFPQ_ARS_0.01%.csv"

# Tiny5m m=16
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[false];gamma:[0.0];k_factor:[1.0];nprobe:[150,250,350,450]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_IVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m tIVFPQ -b 'ivfpq_index_path:"./tmp/index/tiny5m_ivfpq4096x16.bin";C:4096;m:16;nbits:8' -s 'trim_opened:[true];gamma:[0.57];k_factor:[1.0];nprobe:[100,150,200,250,300,350,400,450,500]' -si "./tmp/index/tiny5m_ivfpq4096x16.bin" -sr "./results/QPSDCRecall_Tiny5m_tIVFPQ_ARS_0.01%.csv"
