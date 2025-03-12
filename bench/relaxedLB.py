import math
import numpy as np
import utils
import faiss
from scipy.spatial.distance import cdist

def generate_Unity_landmarks(indexPath, sampled_data_idx):
    # Load PQ index
    index_pq = faiss.read_index(indexPath)  
    # Get PQ codes for the sampled data
    m = index_pq.pq.M 
    n = index_pq.ntotal
    codes = faiss.vector_to_array(index_pq.codes).reshape(n, m)
    sampled_codes = codes[sampled_data_idx]
    # Decode the sampled PQ codes to get landmarks
    landmarks = index_pq.pq.decode(sampled_codes)

    return landmarks

# Load dataset
# dataPath = "../../yitong/Datasets/nytimes-256.hdf5"
# indexPath = "./tmp/index/nytimes_pq8x32.bin"
dataPath = "../../yitong/Datasets/glove-100.hdf5"
indexPath = "./tmp/index/glove_pq8x25.bin"
data, query = utils.read_hdf5_dataset(dataPath, ["train", "test"])
n, d = data.shape

W = 256
sample_num = 10000
gamma = 0.6 # 0.79 for NYTimes, 0.6 for GloVe
sampled_data_idx = np.random.choice(n, size=sample_num, replace=False)

landmarks = generate_Unity_landmarks(indexPath, sampled_data_idx)
print("Generate Unity landmarks done")

strict_distance_ratios = []
pRelaxed_distance_ratios = []

for i in range(len(sampled_data_idx)):
    cur_data_idx = sampled_data_idx[i]
    cur_data = data[cur_data_idx].reshape(1, -1)
    query_idx = np.random.choice(len(query))
    q = query[query_idx].reshape(1, -1)
    
    # real distance
    real_distance = cdist(cur_data, q, metric='euclidean')[0][0]
    
    landmark = landmarks[i].reshape(1, -1)
    landmark_to_data = cdist(cur_data, landmark, metric='euclidean')[0][0]
    landmark_to_query = cdist(q, landmark, metric='euclidean')[0][0]
    
    strictLB = abs(landmark_to_data - landmark_to_query)
    relaxedLB = math.sqrt((landmark_to_data - landmark_to_query)**2 + 2 * gamma * landmark_to_data * landmark_to_query)
    strict_distance_ratios.append(strictLB / real_distance)
    pRelaxed_distance_ratios.append(relaxedLB / real_distance)


print("Strict_distance_ratio:", np.mean(strict_distance_ratios))
print("pRelaxed_distance_ratio:", np.mean(pRelaxed_distance_ratios))
