## Preprocessing
```bash
python3 preprocessing.py -s /data/home/petrizhang/data/vector/raw/nytimes-256-angular.hdf5 -t /data/home/petrizhang/data/vector/normed/nytimes-256-angular.hdf5 
```

## HNSWLIB
```bash
# sift
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_hnswlib16x500.csv
```