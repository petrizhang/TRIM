#!/usr/bin/env bash

set -e

cleanup() {
    exit 1 
}

trap cleanup SIGINT

## HNSW (UNITY Implementation)
# gist
python3 bench.py -k 10 -d ./tmp/data/gist-960-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/gist_unity_hnswlib16x500.bin\";M:16;efConstruction:500" -s "enable_batch_dco:[false];ef:[120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,450,500,550,600,650,700,750,800,900,1000,2000]"  -si ./tmp/index/gist_hnsw16x500.empty -sr ./tmp/results/gist_hnsw16x500.csv

# nytimes
python3 bench.py -k 10 -d ./tmp/data/nytimes-256-angular.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500" -s "enable_batch_dco:[false];ef:[100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,300,4000]"  -si ./tmp/index/nytimes_hnsw16x500.empty -sr ./tmp/results/nytimes_hnsw16x500.csv

## UNITY
# gist
python3 bench.py -k 10 -d ./tmp/data/gist-960-euclidean.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/gist_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/gist_pq8x240.bin\";pq_m:32;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[false,true];ef:[120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,450,500,550,600,650,700,750,800,900,1000,2000]""  -si ./tmp/index/gist_unity_hnsw16x500_pq8x240.bin -sr ./tmp/results/gist_unity_hnsw16x500_pq8x240.csv

# nytimes PQ8x64
python3 bench.py -k 10 -d ./tmp/data/nytimes-256-angular.hdf5 -m unity -b "hnswlib_index_path:\"./tmp/index/nytimes_hnswlib16x500.bin\";M:16;efConstruction:500;pq_index_path:\"./tmp/index/nytimes_pq8x64.bin\";pq_m:64;pq_nbits:8;dco:\"unity\"" -s "enable_batch_dco:[true];gamma:[0.8,0.9,1.0,1.1];ef:[100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,300,4000]" -si ./tmp/index/nytimes_unity_hnsw16x500_pq8x64.empty -sr ./tmp/results/nytimes_unity_hnsw16x500_pq8x64.csv

## Plot
python3 plot.py