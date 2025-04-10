import os

import faiss
import numpy as np
import utils
from scipy.spatial.distance import cdist


def generate_random_landmarks(data, W):

    n = len(data)

    sample_indices = np.random.choice(n, size=W, replace=False)
    
    return data[sample_indices]

def generate_distancing_landmarks(data, W):
    # landmarks_file = "./tmp/index/distancing_landmarks_nytimes.txt"
    landmarks_file = "./tmp/index/distancing_landmarks_glove.txt"
    if os.path.exists(landmarks_file):
        landmarks = np.genfromtxt(landmarks_file, delimiter=',')
        print(f"Loaded landmarks from {landmarks_file}")

    else:
        N, D = data.shape
        if W > N:
            raise ValueError("W cannot exceed the number of samples in the data.")
        
        selected = np.zeros(N, dtype=bool)
        initial_idx = np.random.choice(N)
        selected[initial_idx] = True
        index = faiss.IndexFlatL2(D)
        index.add(data[initial_idx].reshape(1, D))
        
        i = 1
        for _ in range(W - 1):
            candidates_idx = np.where(~selected)[0]
            if len(candidates_idx) == 0:
                break
            
            candidates_data = data[candidates_idx]
            distances, _ = index.search(candidates_data, k=1)
            distances = distances.flatten()
            max_distance_idx = np.argmax(distances)
            next_point_idx = candidates_idx[max_distance_idx]
            
            selected[next_point_idx] = True
            index.add(data[next_point_idx].reshape(1, D))
            if i % 10 == 0:
                print(f"Selecting {i} landmarks...")
            i += 1
        
        landmarks = data[selected]
        np.savetxt(landmarks_file, landmarks, delimiter=',')
        print(f"Saved landmarks to {landmarks_file}")
    
    return landmarks

def generate_clustering_landmarks(data, W, sampled_data_idx):
    
    data = np.array(data).astype(np.float32)
    dim = data.shape[1]
    
    kmeans = faiss.Kmeans(dim, W)
    kmeans.train(data)
    
    centroids = kmeans.centroids
    
    index = faiss.IndexFlatL2(dim)
    index.add(centroids)
    
    sampled_data = data[sampled_data_idx]
    all_distances, all_labels = index.search(sampled_data, 1)
    
    return centroids, all_distances.flatten(), all_labels.flatten()

def generate_trim_landmarks(indexPath, sampled_data_idx):
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
# dataPath = "../../tmp/Datasets/nytimes-256.hdf5"
# indexPath = "./tmp/index/nytimes_pq8x32.bin"
dataPath = "../../tmp/Datasets/glove-100.hdf5"
indexPath = "./tmp/index/glove_pq8x25.bin"
data, query = utils.read_hdf5_dataset(dataPath, ["train", "test"])
n, d = data.shape

W = 256
sample_num = 1000
sampled_data_idx = np.random.choice(n, size=sample_num, replace=False)

random_landmarks = generate_random_landmarks(data, W)
print("Generate random landmarks done")

distancing_landmarks = generate_distancing_landmarks(data, W)
print("Generate distancing landmarks done")

centroids, cluster_landmarks_distances, I = generate_clustering_landmarks(data, W, sampled_data_idx)
print("Generate clustering landmarks done")

trim_landmarks = generate_trim_landmarks(indexPath, sampled_data_idx)
print("Generate trim landmarks done")

distance_ratios1 = []
distance_ratios2 = []
distance_ratios3 = []
distance_ratios4 = []

for i in range(len(sampled_data_idx)):
    cur_data_idx = sampled_data_idx[i]
    cur_data = data[cur_data_idx].reshape(1, -1)
    query_idx = np.random.choice(len(query))
    q = query[query_idx].reshape(1, -1)
    
    # real distance
    real_distance = cdist(cur_data, q, metric='euclidean')[0][0]
    
    # Method 1：random landmarks
    distances_x2l = cdist(cur_data, random_landmarks, metric='euclidean')[0]
    distances_q2l = cdist(q, random_landmarks, metric='euclidean')[0]
    lb_values = abs(distances_x2l - distances_q2l)
    max_lb1 = np.max(lb_values)
    distance_ratios1.append(max_lb1 / real_distance)
    
    # Method 2: distancing landmarks
    distances_x2l = cdist(cur_data, distancing_landmarks, metric='euclidean')[0]
    distances_q2l = cdist(q, distancing_landmarks, metric='euclidean')[0]
    lb_values = abs(distances_x2l - distances_q2l)
    max_lb2 = np.max(lb_values)
    distance_ratios2.append(max_lb2 / real_distance)
    
    # Method 3:clustering landmarks
    cluster_landmark_distance = cluster_landmarks_distances[i]
    cluster_center = centroids[I[i]].reshape(1, -1)
    query_to_centroid = cdist(q, cluster_center, metric='euclidean')[0][0]
    lb3 = abs(cluster_landmark_distance - query_to_centroid)
    distance_ratios3.append(lb3 / real_distance)
    
    # Method 4：trim landmarks
    trim_landmark = trim_landmarks[i].reshape(1, -1)
    landmark_to_data = cdist(cur_data, trim_landmark, metric='euclidean')[0][0]
    landmark_to_query = cdist(q, trim_landmark, metric='euclidean')[0][0]
    lb4 = abs(landmark_to_data - landmark_to_query)
    distance_ratios4.append(lb4 / real_distance)


print("Random Landmarks Ratio:", np.mean(distance_ratios1))
print("Distancing Landmarks Ratio:", np.mean(distance_ratios2))
print("Cluster Landmarks Ratio:", np.mean(distance_ratios3))
print("Trim Landmarks Ratio:", np.mean(distance_ratios4))

