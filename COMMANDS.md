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

python3 bench.py -k 10         -nq 1000         -d "./tmp/data/gist-960-euclidean.hdf5"         -m unity         -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\""         -s "enable_batch_dco:[true];gamma:[0.8,1.0];refine_queue_size:[0,100,200,400,800];ef:[2000]"         -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty"         -sr "./tmp/results/gist_uhnsw16x500_pq8x120.csv"
```

## Testing

### SIFT
```bash
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false,true];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_unity_hnsw16x500_pq8x32.bin -sr ./tmp/results/sift_uhnsw16x500_pq8x32.csv
```


### NYTimes
```bash
python3 bench.py -k 10 -nq 1000 -d "./tmp/data/nytimes-256-angular.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.8];ef:[100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000,5500,6000]" -si "./tmp/index/nytimes_uhnsw16x500_pq8x32.empty" -sr "../results/27b6933_nytimes_uhnsw16x500_pq8x32.csv"
python3 plot.py
```


```bash
python3 bench.py -k 10 -nq 1000 -d "./tmp/data/gist-960-euclidean.hdf5" -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x120.bin\";pq_m:120;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.8,0.802,0.804,0.806,0.81];refine_queue_size:[100,200];ef:[800]" -si "./tmp/index/gist_uhnsw16x500_pq8x120.empty" -sr "../results/5e654e6_nq1000_k10_gist_uhnsw16x500_pq8x120.csv"
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