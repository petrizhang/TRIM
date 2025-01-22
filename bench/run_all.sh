#!/usr/bin/env bash

set -e

version=$(git rev-parse --short HEAD 2>/dev/null || echo "no_version_info")

cleanup() {
    exit 1 
}

trap cleanup SIGINT

# Don't buffer python outputss
export PYTHONUNBUFFERED=1

# Configurations
export nq=1000
export M=16
export efCons=500 

export gist_gamma="[0.8,0.9,1.0,1.1]"
export gist_ef="[120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,450,500,550,600,650,700,750,800,900,1000,2000,3000,4000]"
export gist_pq_m=(30 40 60 80 120 160 240)

export nytimes_gamma="[0.8,0.9,1.0,1.1]"
export nytimes_ef="[100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000]"
export nytimes_pq_m=(8 16 32 64)

#!/bin/bash

bench_hnsw() {
    if [ "$#" -ne 5 ]; then
        echo "no enough parameters"
        return 1
    fi

    local dataset_full_name=$1
    local dataset_short_name=$2
    local M=$3
    local efCons=$4
    local ef=$5

    local command="python3 bench.py -k 10 \
        -nq $nq \
        -d \"./tmp/data/${dataset_full_name}.hdf5\" \
        -m unity \
        -b \"hnswlib_index_path:\\\"./tmp/index/${dataset_short_name}_hnswlib${M}x${efCons}.bin\\\";M:${M};efConstruction:${efCons};dco:\\\"exact\\\"\" \
        -s \"enable_batch_dco:[false];ef:${ef}\" \
        -si \"./tmp/index/${dataset_short_name}_hnsw${M}x${efCons}.empty\" \
        -sr \"../results/${version}_${dataset_short_name}_hnsw${M}x${efCons}.csv\""

    # 执行命令
    echo "Running command: ${command}"
    eval ${command}
}

bench_unity() {
    if [ "$#" -ne 8 ]; then
        echo "no enough parameters"
        return 1
    fi

    local dataset_full_name=$1
    local dataset_short_name=$2
    local M=$3
    local efCons=$4
    local ef=$5
    local dco=$6
    local pq_m=$7
    local gamma=$8

    local command="python3 bench.py -k 10 \
        -nq $nq \
        -d \"./tmp/data/${dataset_full_name}.hdf5\" \
        -m unity \
        -b \"hnswlib_index_path:\\\"./tmp/index/${dataset_short_name}_hnswlib${M}x${efCons}.bin\\\";M:${M};efConstruction:${efCons};pq_index_path:\\\"./tmp/index/${dataset_short_name}_pq8x${pq_m}.bin\\\";pq_m:${pq_m};pq_nbits:8;dco:\\\"$dco\\\"\" \
        -s \"enable_batch_dco:[true];gamma:${gamma};ef:${ef}\" \
        -si \"./tmp/index/${dataset_short_name}_uhnsw${M}x${efCons}_pq8x${pq_m}.empty\" \
        -sr \"../results/${version}_${dataset_short_name}_uhnsw${M}x${efCons}_pq8x${pq_m}.csv\""

    # 执行命令
    echo "Running command: ${command}"
    eval ${command}
}

#################################################
# GIST
#################################################

# HNSW (UNITY Implementation)
bench_hnsw gist-960-euclidean gist $M $efCons $gist_ef

# UNITY
for pq_m in "${gist_pq_m[@]}"; do
    bench_unity gist-960-euclidean gist $M $efCons $gist_ef \
               unity $pq_m $gist_gamma
done

#################################################
# NYTimes
#################################################

# HNSW (UNITY Implementation)
bench_hnsw nytimes-256-angular nytimes $M $efCons $nytimes_ef

## UNITY 
for pq_m in "${nytimes_pq_m[@]}"; do
    bench_unity nytimes-256-angular nytimes $M $efCons $nytimes_ef \
                unity $pq_m $nytimes_gamma
done

## Plot
python3 plot.py