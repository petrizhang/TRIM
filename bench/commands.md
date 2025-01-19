
```bash
python3 bench.py -k 10 -d ./tmp/data/sift-128-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_hnswlib16x500_k10.csv
python3 bench.py -k 100 -d ./tmp/data/sift-128-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[100,120,140,160,200,300,400,500,600,700,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_hnswlib16x500_k100.csv


python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m unity -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_unity_raw_hnsw16x500_k10.csv
python3 bench.py -k 10 -nq 1000 -d ./tmp/data/sift-128-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/sift_hnswlib16x500.bin -sr ./tmp/results/sift_hnsw16x500_k10.csv

python3 bench.py -k 10 -nq 100 -d ./tmp/data/gist-960-euclidean.hdf5 -m hnsw -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/gist_hnswlib16x500.bin -sr ./tmp/results/gist_tophnsw16x500_k10.csv
python3 bench.py -k 10 -nq 100 -d ./tmp/data/gist-960-euclidean.hdf5 -m unity -b "M:16;efConstruction:500" -s "ef:[10,20,30,40,50,60,70,80,90,100,200,400,800]"  -si ./tmp/index/gist_hnswlib16x500.bin -sr ./tmp/results/gist_tophnsw16x500_k10.csv
```