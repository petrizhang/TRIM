## Preprocessing
```bash
python3 preprocessing.py -s /data/home/petrizhang/data/vector/raw/nytimes-256-angular.hdf5 -t /data/home/petrizhang/data/vector/normed/nytimes-256-angular.hdf5 
```

## HNSWLIB
```bash
# sift
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_hnswlib16x500.csv
```

## HNSW (UNITY Implementation)
```bash
# sift
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift_hnswlib16x500.bin\";M:16;efConstruction:500" -s "enable_batch_dco:[false,true];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_unity_hnsw16x500.bin -sr ./tmp/results/sift_unity_hnsw16x500.csv

# gist
python3 bench.py -k 10 -nq 500 -d ./tmp/data/gist-960-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500" -s "enable_batch_dco:[false,true];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800,1000,2000,4000]"  -si ./tmp/index/gist_unity_hnsw16x500.bin -sr ./tmp/results/gist_unity_hnsw16x500.csv

# nytimes
python3 bench.py -k 10 -nq 500 -d ./tmp/data/nytimes-256-angular.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_unity_hnswlib16x500.bin\";M:16;efConstruction:500" -s "enable_batch_dco:[false,true];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800,1000,2000,4000]"  -si ./tmp/index/nytimes_unity_hnsw16x500.bin -sr ./tmp/results/nytimes_unity_hnsw16x500.csv
```

## UNITY

```bash
# sift
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/sift_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/sift_pq8x32.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false,true];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_unity_hnsw16x500_pq8x32.bin -sr ./tmp/results/sift_unity_hnsw16x500_pq8x32.csv

# gist
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/gist-960-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x240.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false,true];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800,1000,2000,4000]"  -si ./tmp/index/gist_unity_hnsw16x500_pq8x240.bin -sr ./tmp/results/gist_unity_hnsw16x500_pq8x240.csv

# nytimes
python3 bench.py -k 10 -nq 500 -d ./tmp/data/nytimes-256-angular.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_unity_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.8,1.0,1.1];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800,1000,2000,4000]" -si ./tmp/index/nytimes_unity_hnsw16x500_pq8x64.bin -sr ./tmp/results/nytimes_unity_hnsw16x500_pq8x64.csv

python3 bench.py -k 10 -nq 500 -d ./tmp/data/nytimes-256-angular.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_unity_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[1.2];ef:[10,20,30,40,50,60,70,80,90,100,200,400,800,1000,2000,4000]" -si ./tmp/index/nytimes_unity_hnsw16x500_pq8x64.bin -sr ./tmp/results/nytimes_unity_hnsw16x500_pq8x64.csv
```