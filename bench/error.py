import math

import faiss
import numpy as np
import utils
from scipy.spatial.distance import cdist


def generate_trim_landmarks(indexPath):
    # Load PQ index
    index_pq = faiss.read_index(indexPath)  
    # Get PQ codes for the sampled data
    m = index_pq.pq.M 
    n = index_pq.ntotal
    codes = faiss.vector_to_array(index_pq.codes).reshape(n, m)
    # Decode the sampled PQ codes to get landmarks
    landmarks = index_pq.pq.decode(codes)

    return landmarks

# Load dataset
# dataPath = "../../tmp/Datasets/nytimes-256.hdf5"
# indexPath = "./tmp/index/nytimes_pq8x32.bin"
# dataPath = "../../tmp/Datasets/gist-960.hdf5"
# indexPath = "./tmp/index/gist_pq8x120.bin"
dataPath = "../../tmp/Datasets/glove-100.hdf5"
indexPath = "./tmp/index/glove_pq8x25.bin"
data, query = utils.read_hdf5_dataset(dataPath, ["train", "test"])
n, d = data.shape

sample_num = 10000
sampled_data_idx = np.random.choice(n, size=sample_num, replace=False)

landmarks = generate_trim_landmarks(indexPath)
print("Generate trim landmarks done")

gammas =  [0.5,0.6,0.7,0.8,0.9,1]

for gamma in gammas:
    
    errors = []

    for i in range(len(sampled_data_idx)):
        cur_data_idx = sampled_data_idx[i]
        cur_data = data[cur_data_idx].reshape(1, -1)
        query_idx = np.random.choice(len(query))
        q = query[query_idx].reshape(1, -1)
        
        real_distance = cdist(cur_data, q, metric='euclidean')[0][0]
        
        landmark = landmarks[cur_data_idx].reshape(1, -1)
        landmark_to_data = cdist(cur_data, landmark, metric='euclidean')[0][0]
        landmark_to_query = cdist(q, landmark, metric='euclidean')[0][0]
        
        relaxedLB = math.sqrt((landmark_to_data - landmark_to_query)**2 + 2 * gamma * landmark_to_data * landmark_to_query)
        errors.append(relaxedLB - real_distance)
        
    errors = np.array(errors)

    error_upper_bound = np.max(errors)                
    error_lower_bound = np.min(errors)               
    error_upper_quartile = np.percentile(errors, 75)
    error_lower_quartile = np.percentile(errors, 25)
    error_median = np.median(errors)

    # print(f"gamma: {gamma}")
    # print(f"Error Upper Bound: {error_upper_bound}")
    # print(f"Error Lower Bound: {error_lower_bound}")
    # print(f"Error Upper Quartile (75%): {error_upper_quartile}")
    # print(f"Error Lower Quartile (25%): {error_lower_quartile}")
    # print(f"Error Median: {error_median}")
    print(f"{gamma},{error_upper_bound},{error_lower_bound},{error_upper_quartile},{error_lower_quartile},{error_median}")