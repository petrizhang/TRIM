import os
import time

import faiss
import numpy as np

# --------------------------------------------------
# 1. 造数据
# --------------------------------------------------
d = 128                # 维度
nb = 128*128        # 数据库大小
nq = 1000              # 查询集大小
np.random.seed(42)

xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.       # 制造一点“顺序”方便观察
xq = np.random.random((nq, d)).astype('float32')

# 计算 ground truth
gt_index = faiss.IndexFlatL2(d)
gt_index.add(xb)
D, gt_I = gt_index.search(xq, 10)

# --------------------------------------------------
# 2. 用 index_factory 建索引
#    IVF1024,PQ16x4fs,RFlat
# --------------------------------------------------
# 1024 个 IVF 簇，PQ 16 段 4bit，FastScan，RefineFlat
index_str = 'IVF128,PQ64x4fs,RFlat'
index = faiss.index_factory(d, index_str, faiss.METRIC_L2)

# 训练
assert not index.is_trained
index.train(xb)
index.add(xb)

# --------------------------------------------------
# 3. 多次调整参数并搜索
# --------------------------------------------------


def evaluate(index, nprobe, k_factor, k=10):
    """返回 recall@k 与 QPS"""
    # 3.1 设置 IVF 的 nprobe
    base_index = faiss.downcast_index(index.base_index)
    base_index.nprobe = nprobe
    index.k_factor = k_factor

    # 3.4 搜索并计时
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    qps = nq / (t1 - t0)

    # 3.5 计算 recall@k
    # 计算 recall@k
    recalls = []
    for gt_ids, pred_ids in zip(gt_I[:, :k], I):
        recalls.append(len(set(gt_ids) & set(pred_ids)) / k)
    recall = np.mean(recalls)
    return recall, qps


print("nprobe,qbs,k_factor,Recall@10,QPS")


for nprobe in [1, 4, 16, 64, 128]:
    for kf in [1, 4, 16, 64]:
        recall, qps = evaluate(index, nprobe, kf, k=10)
        print(f"{nprobe},{kf},{recall:.4f},{qps:.1f}")
