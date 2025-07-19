

## GLOVE
```bash
python3 bench.py -qt ann -k 10 -nq 500 -d "/home/yitong/yitong/Datasets/glove-100.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"/home/yitong/TOP/bench/tmp/index/glove_tivfpqfs4096x50.bin";C:4096;m:50;nbits:4' -s 'gamma:[0.6];nprobe:[60,100,200,400,600,800,1000,1200,1400,1600]' -si "/home/yitong/TOP/bench/tmp/index/glove_tivfpqfs4096x50.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQfs_KNN_k10_avx2.csv"

python3 bench.py -qt ann -k 10 -nq 500 -d "/home/yitong/yitong/Datasets/glove-100.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"/home/yitong/TOP/bench/tmp/index/glove_tivfpqfs4096x50.bin";C:4096;m:50;nbits:4' -s 'gamma:[0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6];nprobe:[60,100,200,400,600,800,1000,1200,1400,1600]' -si "/home/yitong/TOP/bench/tmp/index/glove_tivfpqfs4096x50.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQfs_KNN_k10_avx2.csv"


python3 bench.py -qt ann -k 10 -nq 500 -d "/home/yitong/yitong/Datasets/glove-100.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"/home/yitong/TOP/bench/tmp/index/glove_tivfpqfs4096x50.bin";C:4096;m:50;nbits:4' -s 'gamma:[0.6];nprobe:[60,100,200,400,600,800,1000,1200,1400,1600]' -si "/home/yitong/TOP/bench/tmp/index/glove_tivfpqfs4096x50.bin" -sr "./results/QPSDCRecall_GloVe_tIVFPQfs_KNN_k10_avx512.csv"
```

## GIST

```bash
python3 bench.py -qt ann -k 10 -nq 500 -d "/home/yitong/yitong/Datasets/gist-960.hdf5" -m tIVFPQfs -b 'ivfpqfs_index_path:"/home/yitong/TOP/bench//tmp/index/gist_tivfpqfs4096x480.bin";C:4096;m:480;nbits:4' -s 'gamma:[0.6,0.65,0.68,0.7,0.72,0.75,0.8];nprobe:[40,50,60,70,80,90,100,200,300,500]' -si "/home/yitong/TOP/bench//tmp/index/gist_tivfpqfs4096x480.bin" -sr "./results/QPSDCRecall_GIST_tIVFPQfs_KNN_k10_avx512.csv"
```