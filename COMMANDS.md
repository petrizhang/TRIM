## Preprocessing
```bash
python3 preprocessing.py -s /data/home/petrizhang/data/vector/raw/nytimes-256-angular.hdf5 -t /data/home/petrizhang/data/vector/normed/nytimes-256-angular.hdf5 

python3 preprocessing.py -s /data/home/petrizhang/data/vector/raw/glove-100-angular.hdf5 -t /data/home/petrizhang/data/vector/normed/glove-100-angular.hdf5 

```

## HNSWLIB
```bash
# sift
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_hnswlib16x500.csv
```

## Perf

### GIST
```bash
# HNSW
perf record -g -F 999 python3 bench.py -k 10         -nq 1000         -d "./tmp/data/gist-960-euclidean.hdf5"         -m unity         -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;dco:\"exact\""         -s "enable_batch_dco:[false];ef:[2000,2000,2000,2000]"         -si "./tmp/index/gist_hnsw16x500.empty"         -sr "./tmp/results/51a08bf_gist_hnsw16x500.csv"

# uHNSW
perf record -g -F 999 -p $(pgrep python3)
python3 bench.py -k 10         -nq 1000         -d "./tmp/data/gist-960-euclidean.hdf5"         -m unity         -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\""         -s "enable_batch_dco:[true];gamma:[1.1];ef:[2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000]"         -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty"         -sr "./tmp/results/51a08bf_gist_uhnsw16x500_pq8x120.csv"

python3 bench.py -k 10         -nq 1000         -d "./tmp/data/gist-960-euclidean.hdf5"         -m unity         -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\""         -s "enable_batch_dco:[true];gamma:[0.8];refine_queue_size:[100];ef:[2000,3000]"         -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty"         -sr "./tmp/results/gist_uhnsw16x500_pq8x120.csv"
```

#### Scalability
### 2m

python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift2m.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift2_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift2_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false];gamma:[0.8];ef:[30,50,100,200,300]" -si ./tmp/index/sift2_hnsw16x500_pq8x32.empty -sr ./results/sift_hnsw16x500_pq8x32.csv

### 4m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift4m.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift4_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift4_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false];gamma:[0.8];ef:[30,50,100,200,300,400]" -si ./tmp/index/sift4_hnsw16x500_pq8x32.empty -sr ./results/sift_hnsw16x500_pq8x32.csv

### 6m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift6m.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift6_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift6_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false];gamma:[0.8];ef:[1000]" -si ./tmp/index/sift6_hnsw16x500_pq8x32.empty -sr ./results/sift_hnsw16x500_pq8x32.csv

### 8m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift8m.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift8_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift8_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false];gamma:[0.8];ef:[1000]" -si ./tmp/index/sift8_hnsw16x500_pq8x32.empty -sr ./results/sift_hnsw16x500_pq8x32.csv

### 10m
python3 scalability.py -qt ann -k 10 -nq 1000 -d ../../yitong/Datasets/sift10m/sift10m.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift10_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift10_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false];gamma:[0.8];ef:[1000]" -si ./tmp/index/sift10_hnsw16x500_pq8x32.empty -sr ./results/sift_hnsw16x500_pq8x32.csv

# parameter p and lambda
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.67,0.88,0.89,0.9,0.95,0.97,0.99];ef:[1000]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "./results/gist_uhnsw16x500_pq8x120.csv"


python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.79,0.89,0.92,0.95,0.97,0.99,1.0];ef:[1000]" -si "./tmp/index/nytimes_uhnsw16x500_pq8x32.empty" -sr "./results/nytimes_uhnsw16x500_pq8x120.csv"

##### HNSW-ann

# GIST
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_uHNSW_KNN_k10.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[3500,4000]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_uHNSW_KNN_k100.csv"
<!-- 100,200,300,400,600,800,1000,1200,1400,1600,1800,2000,2500,3000 -->

# GloVe
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.61];ef:[10,30,50,70,90,150,300,400,600,800,1000,1500,2000,3000,4000]" -si "./tmp/index/glove_uhnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_uHNSW_KNN_k10.csv"

python3 bench.py -qt ann -k 100 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[100,200,400,600,800,1000,1200,1500]" -si "./tmp/index/glove_uhnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_uHNSW_KNN_k100.csv"

# NYTimes
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.76];ef:[10,50,90,150,300,400,600,800,1000,3000,5000]" -si "./tmp/index/nytimes_uhnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_uHNSW_KNN_k10.csv"

# Tiny5m
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[10,50,90,150,300,600,800,1000,2000,3000,4000]" -si "./tmp/index/tiny5m_uhnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_uHNSW_KNN_k10.csv"


#### HNSW-range
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.67];ef:[10,30,50,70,90,150,300,400,600,800,1000]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_uHNSW_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.7];ef:[350,400,450,500,550,600,650,700,800,900,1000]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "./results/QPSDCRecall_GIST_uHNSW_ARS_0.1%.csv"

# GloVe
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.63];ef:[5500,6000]" -si "./tmp/index/glove_uhnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_uHNSW_ARS_0.01%.csv"
<!-- 50,70,90,100,200,400,600,800,1000,1200,1500,2000,2500,3000,3500,4000,5000 -->

python3 bench.py -qt range -se 0.1 -nq 1000 -d "../../yitong/Datasets/glove-100.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/glove_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/glove_pq8x25.bin\";pq_m:25;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.65];ef:[900,950,1000]" -si "./tmp/index/glove_uhnsw16x500_pq8x25.empty" -sr "./results/QPSDCRecall_GloVe_uHNSW_ARS_0.1%.csv"

# NYTimes
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/nytimes-256.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.75];ef:[30,50,100,150,300,400,600,800,1000,3000,5000,7000]" -si "./tmp/index/nytimes_uhnsw16x500_pq8x64.empty" -sr "./results/QPSDCRecall_NYTimes_uHNSW_ARS_0.01%.csv"

# Tiny5m
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/tiny5m-384.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/tiny5m_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/tiny5m_pq8x96.bin\";pq_m:96;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.64];ef:[100,120,150,200,250,300,400,600,800,1000,2000,3000,4000]" -si "./tmp/index/tiny5m_uhnsw16x500_pq8x96.empty" -sr "./results/QPSDCRecall_Tiny5m_uHNSW_ARS_0.01%.csv"


#### IVFPQ-knn
# GIST
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m uIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'unity_opened:[false];gamma:[0.67];k_factor:[50.0];nprobe:[40,60,80,100,150,200,250,300]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m uIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'unity_opened:[true];gamma:[0.67];k_factor:[50.0];nprobe:[40,60,80,100,150,200,250,300]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/QPSDCRecall_GIST_uIVFPQ_KNN_k10.csv"

# GloVe
python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/glove-960.hdf5" -m uIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'unity_opened:[false];gamma:[0.67];k_factor:[50.0];nprobe:[40,60,80,100,150,200,250,300]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_KNN_k10.csv"

python3 bench.py -qt ann -k 10 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m uIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'unity_opened:[true];gamma:[0.67];k_factor:[50.0];nprobe:[40,60,80,100,150,200,250,300]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/QPSDCRecall_GIST_uIVFPQ_KNN_k10.csv"

# IVFPQ-range
python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m uIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'unity_opened:[false];gamma:[0.67];k_factor:[1.5];nprobe:[40,60,80,100,150,200,250,300,400]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/QPSDCRecall_GIST_IVFPQ_ARS_0.01%.csv"

python3 bench.py -qt range -se 0.01 -nq 1000 -d "../../yitong/Datasets/gist-960.hdf5" -m uIVFPQ -b 'ivfpq_index_path:"./tmp/index/gist_ivfpq4096x120.bin";C:4096;m:120;nbits:8' -s 'unity_opened:[true];gamma:[0.67];k_factor:[1.0];nprobe:[40,60,80,100,150,200,250,300,400]' -si "./tmp/index/gist_ivfpq4096x120.bin" -sr "./results/QPSDCRecall_GIST_uIVFPQ_ARS_0.01%.csv"


### NYTimes
```bash
python3 bench.py -k 10 -nq 1000 -d "./tmp/data/nytimes-256-angular.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.8];ef:[100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000,5500,6000]" -si "./tmp/index/nytimes_uhnsw16x500_pq8x32.empty" -sr "../results/27b6933_nytimes_uhnsw16x500_pq8x32.csv"
python3 plot.py
```


```bash
python3 bench.py -k 10 -nq 1000 -d "./tmp/data/gist-960-euclidean.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.8,0.9];refine_queue_size:[100,200];ef:[800]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "../results/main_nq1000_k10_gist_uhnsw16x500_pq8x120.csv"


python3 bench.py -k 10 -nq 1000 -d "./tmp/data/gist-960-euclidean.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_opq8x120.bin\";use_opq:true;pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0,0.8];refine_queue_size:[0];ef:[800]" -si "./tmp/index/gist_uhnsw16x500_opq8x120.empty" -sr "../results/main_nq1000_k10_gist_uhnsw16x500_opq8x120.csv"

python3 bench.py -k 10 -nq 1000 -d "./tmp/data/nytimes-256-angular.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.85,0.86,0.87,0.88];refine_queue_size:[10,20,30,40,50,60,70,80];ef:[10,20,30,40,50,60,70,80]" -si "./tmp/index/nytimes_uhnsw16x500_pq8x32.empty" -sr "../results/4c14a6f_nq10_k10_nytimes_uhnsw16x500_pq8x32.csv"
```



```bash
# 遍历当前目录下所有以 .cpp 结尾的文件
for file in *.cpp
do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        echo "#include \"faiss/$file\""  # 打印所需格式
    fi
done


```