import heapq
import sys
import csv

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import faiss
import os
import math
from collections import defaultdict
from scipy.spatial.distance import cdist


class LogContext:
    def __init__(self, info):
        self.info = info

    def __enter__(self):
        # 进入上下文时打印信息
        print(f"Start {self.info} ...")
        return self  # 可以返回一个对象，该对象将作为上下文的返回值

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时打印相同的信息
        print(f"Done {self.info}.")
        # 如果需要处理异常，可以在这里进行
        # 如果返回 False，则异常将被传播；如果返回 True，则异常将被忽略
        return False  # 通常我们希望异常被传播，所以返回 False


def euclidean_distance_matrix(A, B):
    # 计算A中每个向量与B中每个向量的欧几里得距离
    # A和B是两个2维的NumPy数组，其中每一行代表一个向量
    # 首先计算A中每个向量与B中每个向量的差的平方
    diff_squared = np.square(A[:, np.newaxis, :] - B[np.newaxis, :, :])

    # 然后对每个差值的平方求和，得到距离的平方
    dist_squared = np.sum(diff_squared, axis=-1)

    # 最后对距离的平方求平方根，得到欧式距离
    distances = np.sqrt(dist_squared)

    return distances

def euclidean_distance_matrix_parallel(X, Y, batch_size_x=4096, batch_size_y=4096, n_jobs=-1):
    n, d = X.shape
    m, _ = Y.shape
    distance_matrix = np.zeros((n, m))

    def compute_partial_distance_matrix(X_batch_indices, Y_batch_indices):
        # 计算X_batch与Y_batch之间的距离矩阵
        return euclidean_distance_matrix(X[X_batch_indices], Y[Y_batch_indices])

    # 计算X和Y的batch数量
    num_batches_x = (n + batch_size_x - 1) // batch_size_x
    num_batches_y = (m + batch_size_y - 1) // batch_size_y

    # 并行计算每个X_batch与每个Y_batch之间的距离矩阵
    tasks = []
    for i in range(num_batches_x):
        for j in range(num_batches_y):
            X_batch_start = i * batch_size_x
            X_batch_end = min((i + 1) * batch_size_x, n)
            Y_batch_start = j * batch_size_y
            Y_batch_end = min((j + 1) * batch_size_y, m)
            X_batch_indices = np.arange(X_batch_start, X_batch_end)
            Y_batch_indices = np.arange(Y_batch_start, Y_batch_end)
            tasks.append(
                delayed(compute_partial_distance_matrix)(
                    X_batch_indices, Y_batch_indices
                )
            )

    results = Parallel(n_jobs=n_jobs, require="sharedmem")(tasks)

    # 将结果填充到距离矩阵中
    for idx, result in enumerate(results):
        i = idx // num_batches_y  # X的batch索引
        j = idx % num_batches_y  # Y的batch索引
        X_batch_start = i * batch_size_x
        X_batch_end = min((i + 1) * batch_size_x, n)
        Y_batch_start = j * batch_size_y
        Y_batch_end = min((j + 1) * batch_size_y, m)
        distance_matrix[X_batch_start:X_batch_end, Y_batch_start:Y_batch_end] = result

    return distance_matrix

# distance_matrix = compute_pairwise_distances(X, Y)
def greedy_landmark_selection(distance_matrix, k, random_seed=42):
    np.random.seed(random_seed)
    num_points = distance_matrix.shape[0]
    landmarks = []
    landmark_set = set()

    # 随机选择第一个Landmark点
    first_landmark = np.random.randint(num_points)
    landmarks.append(first_landmark)
    landmark_set.add(first_landmark)

    for _ in range(k - 1):
        distances = []
        for i in range(num_points):
            if i in landmark_set:
                distances.append(float("-inf"))
                continue
            min_distance = np.min(distance_matrix[i, landmarks])
            distances.append(min_distance)
        next_landmark = np.argmax(distances)
        landmarks.append(next_landmark)
        landmark_set.add(next_landmark)

    return landmarks

def alt_lowerbound(b2l_distances_all, q2l_distances_all, qi, bi):
    b2l_distances = b2l_distances_all[bi]
    q2l_distances = q2l_distances_all[qi]
    tightest_lowerbound = np.max(np.abs(b2l_distances - q2l_distances))
    return tightest_lowerbound

class Item:
    def __init__(self, id, dist) -> None:
        self.id = id
        self.dist = dist

    def __lt__(self, other):
        if self.dist > other.dist:
            return True
        return False

    def __str__(self) -> str:
        return f"({self.id}, {self.dist :.04f})"

    def __repr__(self) -> str:
        return f"({self.id}, {self.dist :.04f})"


KEY_LOWERBOUND_COMPUTATION1 = "#lowerbound computation"
KEY_LOWERBOUND_COMPUTATION2 = "#lowerbound computation"
KEY_DISTANCE_COMPUTATION1 = "#distance computation (random)"
KEY_DISTANCE_COMPUTATION2 = "#distance computation (Trim)"
GAMMA = "gamma"


def counter_add(d: dict, key: str, n: int = 1):
    count = d.get(key, 0)
    count += n
    d[key] = count


class IndexALT:
    def __init__(self, num_landmarks: int, n_jobs=16) -> None:
        self.num_landmarks = num_landmarks
        self.n_jobs = n_jobs

        self.base = None
        self.dim = None
        
        # random landmarks
        self.randomLandmarks = None
        self.randomLandmark_ids = None
        self.randomLandmark_id_set = None
        self.b2l_dist_table1 = None

        # PQ landmarks
        self.PQLandmarks = None
        self.PQCodes = None
        self.b2l_dist_table2 = None
        self.index_pq = None
        self.pq_m = None
        self.centroids = None

    def build(self, data, dim):
        assert self.base is None
        # 1. add data
        self.base = data
        self.dim = dim

        # 2. generate landmarks
        with LogContext("landmark selection"):
            self._generate_landmarks()

        # 3. compute distance table for base vectors and landmarks
        with LogContext("computing base distance table"):
            self._compute_base_distance_table()

    def knn_search(self, query: np.ndarray, k, gammas):
        assert query.ndim == 2
        nq = query.shape[0]

        result1_id_array = np.zeros(shape=(nq, k), dtype="float32")
        # result1_dist_array = np.zeros(shape=(nq, k), dtype="float32")
        stats1_list = []

        # result2_id_array = np.zeros(shape=(nq, k), dtype="float32")
        # result2_dist_array = np.zeros(shape=(nq, k), dtype="float32")
        stats2_list = []

        # 0. base to query distance table
        with LogContext("computing query to base distance table"):
            b2q_dist_table = euclidean_distance_matrix_parallel(
                self.base, query, n_jobs=self.n_jobs
            )

        # 1. compute query distance table
        with LogContext("computing query to landmark distance table"):
            q2l_dist_table1 = self._compute_query_distance_table(query)

        with LogContext("running queries"):
            for qi, q in enumerate(query):
                stats1 = dict()
                counter_add(stats1, KEY_DISTANCE_COMPUTATION1, self.num_landmarks)

                # 先算random的，获得ground truth
                q2l_dist_list1 = q2l_dist_table1[qi]
                results = [
                    Item(self.randomLandmark_ids[i], q2l_dist_list1[i]) ## landmark也是数据点，先把landmark放进去
                    for i in range(self.num_landmarks)
                ]
                heapq.heapify(results)
                while len(results) > k:
                    heapq.heappop(results)

                for vi, v in enumerate(self.base):
                    if vi in self.randomLandmark_id_set:
                        continue

                    if len(results) < k:
                        dist = b2q_dist_table[vi, qi]
                        counter_add(stats1, KEY_DISTANCE_COMPUTATION1)
                        heapq.heappush(results, Item(vi, dist))
                    else:
                        max_dist = results[0].dist
                        lowerbound = alt_lowerbound(
                            self.b2l_dist_table1, q2l_dist_table1, qi, vi
                        )
                        counter_add(stats1, KEY_LOWERBOUND_COMPUTATION1)
                        if lowerbound > max_dist:
                            continue
                        else:
                            dist = b2q_dist_table[vi, qi]
                            counter_add(stats1, KEY_DISTANCE_COMPUTATION1)
                            heapq.heappush(results, Item(vi, dist))
                            if len(results) > k:
                                heapq.heappop(results)
                
                stats1_list.append(stats1)

                results1 = [
                    (x.id, x.dist) for x in [heapq.heappop(results) for _ in range(k)]
                ]
                results1.reverse()
                result1_id_array = [x[0] for x in results1]
                # a = np.array(results1)
                # result1_id_array[qi, :] = a[:, 0]
                # result1_dist_array[qi, :] = a[:, 1]

                # if self.dim < 4:
                #     continue

                # 算PQ的，并计算recall
                for gamma in gammas:
                    stats2 = {"gamma": gamma}   
                    counter_add(stats2, KEY_DISTANCE_COMPUTATION2, self.index_pq.pq.ksub)
                    results = []
                    heapq.heapify(results)
                    for vi, v in enumerate(self.base):
                        if len(results) < k:
                            dist = b2q_dist_table[vi, qi]
                            counter_add(stats2, KEY_DISTANCE_COMPUTATION2)
                            heapq.heappush(results, Item(vi, dist))
                        else:
                            max_dist = results[0].dist
                            lowerbound = self.alt_lowerboundTRIM(q, vi, gamma)
                            counter_add(stats2, KEY_LOWERBOUND_COMPUTATION2)
                            if lowerbound > max_dist:
                                continue
                            else:
                                dist = b2q_dist_table[vi, qi]
                                counter_add(stats2, KEY_DISTANCE_COMPUTATION2)
                                heapq.heappush(results, Item(vi, dist))
                                if len(results) > k:
                                    heapq.heappop(results)
                    
                    stats2_list.append(stats2)

                    results2 = [
                        (x.id, x.dist) for x in [heapq.heappop(results) for _ in range(k)]
                    ]
                    results2.reverse()
                    # b = np.array(results)
                    result2_id_array = [x[0] for x in results2]
                    
                    # 算recall
                    gt_set = set(result1_id_array)  # ground truth (from randomLandmark method)
                    # print(gt_set)
                    pred_set = set(result2_id_array)  # predicted (from PQ method)
                    # print(pred_set)
                    hit = len(gt_set & pred_set)
                    recall = hit / k
                    stats2["recall"] = recall          

        return pd.DataFrame(stats1_list), pd.DataFrame(stats2_list)

    def _generate_landmarks(self):
        # 从base中随机采样num_landmarks个landmark
        self.randomLandmark_ids = np.random.choice(
            len(self.base), size=self.num_landmarks, replace=False
        )
        self.randomLandmarks = self.base[self.randomLandmark_ids]
        self.randomLandmark_id_set = set(self.randomLandmark_ids)

        # if self.dim < 4:
        #     return
        
        # 用PQ生成landmark
        if self.dim == 2:
            self.pq_m = 2
        else:
            self.pq_m = self.dim // 4  # 一般用整除避免小数
        pq_path = f"/home/yitong/TOP/bench/tmp/index/random_pq8x{self.pq_m}.index"

        # 判断索引是否存在
        if os.path.exists(pq_path):
            print(f"Loading existing PQ index from {pq_path}")
            self.index_pq = faiss.read_index(pq_path)

        else:
            print(f"PQ index not found at {pq_path}, creating new index...")
            print(self.base.shape, self.base.dtype)
            self.index_pq = faiss.IndexPQ(self.dim, self.pq_m, 8) 
            self.index_pq.train(self.base)
            self.index_pq.add(self.base)    
            faiss.write_index(self.index_pq, pq_path)  # 保存索引到磁盘
            print(f"PQ index saved to {pq_path}")
        
        # 获取 PQ 编码并解码 landmark
        self.PQCodes = faiss.vector_to_array(self.index_pq.codes).reshape(self.index_pq.ntotal, self.pq_m)
        self.PQLandmarks = self.index_pq.pq.decode(self.PQCodes)

        # self.PQLandmark_ids = np.arange(index_pq.ntotal)
        # self.PQLandmark_id_set = set(self.PQLandmark_ids)

    def _compute_base_distance_table(self):
        # 随机landmark，存的是距离
        self.b2l_dist_table1 = euclidean_distance_matrix_parallel(
            self.base, self.randomLandmarks, n_jobs=self.n_jobs
        )
        # if self.dim < 4:
        #     return      
        # PQLandmark，存的是距离的平方
        # print(self.base.shape)
        # print(self.PQLandmarks.shape)
        diffs = self.base - self.PQLandmarks
        self.b2l_dist_table2 = np.sum(diffs ** 2, axis=1)


    def _compute_query_distance_table(self, query):
        
        table1 = euclidean_distance_matrix_parallel(
            query, self.randomLandmarks, n_jobs=self.n_jobs
        )

        # pq = self.index_pq.pq
        # M = pq.M
        # Ks = pq.ksub
        # d_sub = self.dim // M

        # self.centroids
        # centroids = np.array(self.index_pq.pq.centroids).reshape(M, Ks, d_sub)

        # distance_tables_all = np.empty((len(query), M, Ks), dtype='float32')
        
        # for i, q in enumerate(query):
        #     tables = np.empty(M * Ks, dtype='float32')
        #     self.index_pq.compute_distance_tables(q, tables)
        #     distance_tables_all[i] = tables.reshape(M, Ks)

        return table1

    def alt_lowerboundTRIM(self, q, vi, gamma):
        b2l_distance = self.b2l_dist_table2[vi]
        b2l_distance = math.sqrt(b2l_distance)
        
        # base_codes = self.PQCodes[bi]
        # q2l_distances = sum(lut[m, base_codes[m]] for m in range(self.pq_m))
        trim_landmark = self.PQLandmarks[vi].reshape(1, -1)
        q = q.reshape(1, -1)
        q2l_distance = cdist(q, trim_landmark, metric='euclidean')[0][0]

        pRLB = math.sqrt((b2l_distance-q2l_distance)**2 + 2*gamma*b2l_distance*q2l_distance)
        return pRLB

def experiment_task(dim, n_landmarks, gammas):
    n_base = 10000
    n_query = 10
    base = np.random.randn(n_base, dim).astype(np.float32)
    query = np.random.randn(n_query, dim).astype(np.float32)
    
    # n_base_samples = min(n_landmarks * 100, 1000)
    k = 10

    index = IndexALT(n_landmarks)
    index.build(base, dim)

    stats1, stats2 = index.knn_search(query, k, gammas)

    # print(stats1)
    prune_ratio1 = 1 - stats1[KEY_DISTANCE_COMPUTATION1].mean() / n_base

    print("Random:")
    print(f"dim:{dim}, prune_ratio:{prune_ratio1:.3f}")

    # if dim >= 4:
    recall_sum = defaultdict(float)
    disCom2_sum = defaultdict(float)
    count = defaultdict(int)

    # print(stats2)
    for _, entry in stats2.iterrows():
        gamma = entry["gamma"]
        recall = entry["recall"]
        disCom2 = entry[KEY_DISTANCE_COMPUTATION2]
        recall_sum[gamma] += recall
        disCom2_sum[gamma] += disCom2
        count[gamma] += 1

    gamma_to_avg_recall = {gamma: recall_sum[gamma] / count[gamma] for gamma in recall_sum}
    gamma_to_disCom2 = {gamma: disCom2_sum[gamma] / count[gamma] for gamma in disCom2_sum}
    prune_ratio2 = {gamma: max(0, 1 - gamma_to_disCom2[gamma]/n_base) for gamma in gamma_to_disCom2}

    print("PQ:")
    
    for gamma in gamma_to_disCom2:
        print(f"dim: {dim}, gamma: {gamma}, prune_ratio: {prune_ratio2[gamma]:.3f}, recall: {gamma_to_avg_recall[gamma]:.3f}")

    # === 写入 CSV ===
    output_path = "/home/yitong/TOP/bench/results/testDim.csv"
    file_exists = os.path.exists(output_path)

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 写标题（如果文件不存在）
        if not file_exists:
            writer.writerow(["dim", "gamma", "prune_ratio", "recall"])

        # 写入 Random 结果（recall 留空）
        writer.writerow([dim, "random", f"{prune_ratio1:.3f}", "1.000"])

        # if dim >= 4:
        # 写入每个 gamma 的 PQ 结果
        for gamma in gamma_to_avg_recall:
            writer.writerow([
                            dim,
                            gamma,
                            f"{prune_ratio2[gamma]:.3f}",
                            f"{gamma_to_avg_recall[gamma]:.3f}"
                        ])


if __name__ == "__main__":

    n_landmarks = 100
    dim = 2
    gammas = [0.0]
    # gammas = [0.16,0.17,0.18,0.19]
    # gammas = [0.5, 0.55, 0.6, 0.65, 0.7,0.75, 0.8, 0.85 ,0.9,0.95, 1]
    experiment_task(dim, n_landmarks, gammas)
    