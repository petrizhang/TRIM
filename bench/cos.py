import numpy as np
import utils
import faiss

def cos_similarity2(X, L, Q):
    X = np.array(X)
    L = np.array(L)
    Q = np.array(Q)
    
    X_minus_L = X - L
    Q_minus_L = Q - L
    
    dot_product = np.dot(X_minus_L, Q_minus_L)
    
    norm_X_minus_L = np.linalg.norm(X_minus_L)
    norm_Q_minus_L = np.linalg.norm(Q_minus_L)
    
    cos_sim = dot_product / (norm_X_minus_L * norm_Q_minus_L)
    
    return cos_sim * cos_sim

def cos_similarity(X, L, Q):
    X = np.array(X)
    L = np.array(L)
    Q = np.array(Q)
    
    X_minus_L = X - L
    Q_minus_L = Q - L
    
    dot_product = np.dot(X_minus_L, Q_minus_L)
    
    norm_X_minus_L = np.linalg.norm(X_minus_L)
    norm_Q_minus_L = np.linalg.norm(Q_minus_L)
    
    cos_sim = dot_product / (norm_X_minus_L * norm_Q_minus_L)
    
    return cos_sim


# Load dataset
dataPath = "../../yitong/Datasets/nytimes-256.hdf5"
indexPath = "./tmp/index/nytimes_pq8x32.bin"
sample_size = 1000  # Number of samples to take

data, query = utils.read_hdf5_dataset(dataPath, ["train", "test"])  # Load data
index_pq = faiss.read_index(indexPath)  # Load PQ index

m = index_pq.pq.M  # Number of subspaces
n = index_pq.ntotal  # Total number of data in index
print(f"Number of subspaces: {m}")
print(f"Number of vectors in index: {n}")

# Randomly sample indices
sample_indices = np.random.choice(n, size=sample_size, replace=False)

# Get PQ codes for the sampled data
codes = faiss.vector_to_array(index_pq.codes).reshape(n, m)
sampled_codes = codes[sample_indices]
print(f"PQ codes shape for sampled data: {sampled_codes.shape}")

# Decode the sampled PQ codes to get landmarks
landmarks = index_pq.pq.decode(sampled_codes)
print(f"Landmarks shape: {landmarks.shape}")
sampled_data = data[sample_indices]
sampled_query = query[:sample_size]
# dim = sampled_data.shape[1]
# sampled_query = np.random.normal(loc=0, scale=1, size=(sample_size, dim))

# cos_values = []
# for X, L, Q in zip(sampled_data, landmarks, sampled_query):
#     cos_values.append(cos_similarity2(X, L, Q))

# sorted = np.sort(cos_values)
# formatted_values = [f"{value:.8f}" for value in sorted]
# print(",".join(map(str, formatted_values)))

cos_values = []
for X, L, Q in zip(sampled_data, landmarks, sampled_query):
    cos_values.append(1-cos_similarity(X, L, Q))

sorted = np.sort(cos_values)

gammas = np.linspace(0, 2, 100)
cdf_values = []
for gamma in gammas:
    count = np.sum(cos_values >= gamma)
    proportion = count / len(cos_values)
    cdf_values.append(proportion)

# 输出
formatted_gammas = [f"{value:.2f}" for value in gammas]
print("gamma = [" + ",".join(map(str, formatted_gammas)) + "]")
formatted_cdfs = [f"{value:.2f}" for value in cdf_values]
print("cdf = [" + ",".join(map(str, formatted_cdfs)) + "]")



