import numpy as np
import utils
import faiss
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import math

def compute_h1_h2(X, L):
    X_minus_L = X - L
    X_minus_L_norm = np.linalg.norm(X_minus_L)
    
    if X_minus_L_norm < 1e-10:
        X_minus_L_norm = 1e-10
    
    dot_product = np.dot(X_minus_L, L)
    h1 = dot_product / X_minus_L_norm
    
    L_norm_squared = np.dot(L, L)
    h1_squared = h1 ** 2
    h2_squared = L_norm_squared - h1_squared
    
    if h2_squared < 0:
        h2_squared = 0
    
    return h1_squared, h2_squared

def cos_similarity(X, L, Q):
    X = np.array(X)
    L = np.array(L)
    Q = np.array(Q)
    
    X_minus_L = X - L
    Q_minus_L = Q - L
    
    dot_product = np.dot(X_minus_L, Q_minus_L)
    
    norm_X_minus_L = np.linalg.norm(X_minus_L)
    norm_Q_minus_L = np.linalg.norm(Q_minus_L)
    
    if norm_X_minus_L * norm_Q_minus_L < 1e-10:
        return 0
    
    return dot_product / (norm_X_minus_L * norm_Q_minus_L)

def compute_gamma_for_Gauss(ps, sampled_data, landmarks, num_samples=1000):
    
    d = len(sampled_data[0])
    if d < 3:
        raise ValueError("Dimension d must be at least 3")
    
    all_gammas = []
    for X, L in zip(sampled_data, landmarks):
        
        h1_squared, h2_squared = compute_h1_h2(X, L)

        A = np.random.noncentral_chisquare(df=1, nonc=h1_squared, size=num_samples)
        B = np.random.noncentral_chisquare(df=1, nonc=h2_squared, size=num_samples)
        C = np.random.chisquare(df=d - 3, size=num_samples)

        if (A + B + C == 0).any():
            raise ValueError("A + B + C contains zero values")
    
        Z_squared = A / (A + B + C)
        valid_indices = ~np.isnan(Z_squared) & (Z_squared >= 0)
        Z_squared = Z_squared[valid_indices]
        if len(Z_squared) == 0:
            raise ValueError("All values in Z_squared are NaN or invalid. Check the input data and calculations.")
        
        sorted_Z_squared = np.sort(Z_squared)
        ecdf_Z_squared = np.arange(1, len(sorted_Z_squared)+1) / len(sorted_Z_squared)
    
        F_Z_squared = interp1d(sorted_Z_squared, ecdf_Z_squared, kind='cubic', fill_value=(0, 1), bounds_error=False)
    
        y_values = np.linspace(0, 2, 1000)
        F_1Z_values = []
        for y in y_values:
            value = (1 - y)**2
            prob_Z = F_Z_squared(value)
            if prob_Z < 0 :  prob_Z = 0
            elif prob_Z > 1 :  prob_Z = 1
            if y <= 1:
                F_1Z = 0.5 - 0.5 * prob_Z  
            else:
                F_1Z = 0.5 + 0.5 * prob_Z
            F_1Z_values.append(1-F_1Z)
        
        F_1Z_values = np.array(F_1Z_values)

        # formatted_gammas = [f"{value:.2f}" for value in y_values]
        # print("gamma = [" + ",".join(map(str, formatted_gammas)) + "]")
        # formatted_cdfs = [f"{value:.2f}" for value in F_1Z_values]
        # print("cdf = [" + ",".join(map(str, formatted_cdfs)) + "]")
        
        gammas = []
        for p in ps:
            abs_diff = np.abs(F_1Z_values - p)
            min_diff_idx = np.argmin(abs_diff)
            closest_indices = np.where(abs_diff == abs_diff[min_diff_idx])[0]
            max_closest_idx = np.max(closest_indices)
            gammas.append(y_values[max_closest_idx])
        
        all_gammas.append(gammas)

    all_gammas = np.array(all_gammas)
    # gamma_min = np.min(all_gammas, axis=0)  # minimal value
    gamma_q1 = np.percentile(all_gammas, 1, axis=0)  # lower quartile
    # gamma_means= np.mean(all_gammas, axis=0) # mean
    return gamma_q1

def compute_gamma_for_others(ps, sampled_data, landmarks, queries):
    
    all_gammas = []
    for Q in queries:
        cos_values = []
        for X, L in zip(sampled_data, landmarks):
            cos_values.append(1-cos_similarity(X, L, Q))

        cos_values = np.sort(cos_values)
        
        cdf_values = np.array([np.sum(cos_values >= gamma) / len(cos_values) for gamma in cos_values])

        unique_cdf, unique_gamma = [], []
        used_cdf = set()
        for gamma, cdf in zip(cos_values[::-1], cdf_values[::-1]):
            if cdf not in used_cdf:
                unique_cdf.append(cdf)
                unique_gamma.append(gamma)
                used_cdf.add(cdf)

        unique_cdf = np.array(unique_cdf[::-1])
        unique_gamma = np.array(unique_gamma[::-1])

        order = np.argsort(unique_cdf)
        unique_cdf = unique_cdf[order]
        unique_gamma = unique_gamma[order]

        gamma_from_cdf = interp1d(unique_cdf, unique_gamma, kind='cubic', fill_value=(cos_values[0], cos_values[-1]), bounds_error=False)

        gamma_values = gamma_from_cdf(ps)
        all_gammas.append(gamma_values)
    
    all_gammas = np.array(all_gammas)
    gamma_min = np.min(all_gammas, axis=0)  # minimal value
    # gamma_q1 = np.percentile(all_gammas, 0.5, axis=0)  # lower quartile
    # gamma_means= np.mean(all_gammas, axis=0) # mean

    return gamma_min

def validate(gamma, sampled_data, landmarks, queries):

    for Q in queries:
        num = 0
        wrong = 0
        Q = Q.reshape(1, -1)
        for X, L in zip(sampled_data, landmarks):
            X = X.reshape(1, -1)
            L = L.reshape(1, -1)
            num = num + 1
            real_dist = cdist(X, Q, metric='euclidean')[0][0]
            real_dist = real_dist**2
            landmark_to_data = cdist(X, L, metric='euclidean')[0][0]
            landmark_to_query = cdist(Q, L, metric='euclidean')[0][0]
            lowerbound = (landmark_to_data - landmark_to_query)**2 + 2 * gamma * landmark_to_data * landmark_to_query
            if lowerbound > real_dist:
                wrong = wrong + 1
        print(f"Total: {num}, Wrong: {wrong}, Ratio: {(wrong / num * 100):.3f}%")

# Load dataset
dataPath = "../../yitong/Datasets/nytimes-256.hdf5"
indexPath = "./tmp/index/nytimes_pq8x64.bin"
sample_size = 3000  # Number of samples to take

# dataPath = "../../yitong/Datasets/gist-960.hdf5"
# indexPath = "./tmp/index/gist_pq8x240.bin"
# sample_size = 1000  # Number of samples to take

# dataPath = "../../yitong/Datasets/glove-100.hdf5"
# indexPath = "./tmp/index/glove_pq8x25.bin"
# sample_size = 1000  # Number of samples to take

# dataPath = "../../yitong/Datasets/tiny5m-384.hdf5"
# indexPath = "./tmp/index/tiny5m_pq8x96.bin"
# sample_size = 10000  # Number of samples to take

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
sampled_data = data[sample_indices]
query = query[:1000]

# if len(query) < sample_size:
#     num_to_fill = sample_size - len(query)
#     fill_indices = np.random.choice(len(query), size=num_to_fill, replace=True)
#     fill_data = query[fill_indices]
#     sampled_query = np.concatenate((query, fill_data))
# else:
#     sampled_query = query[:sample_size]

print(f"Sampled_data shape: {sampled_data.shape}")
print(f"Landmarks shape: {landmarks.shape}")
print(f"Query shape: {query.shape}")

# Prepare p values
# p_values = [0.5,0.6,0.7,0.8,0.9,0.9999]
p_values = [0.91,0.93,0.95,0.97,0.999]
# p_values = [0.8,0.85,0.9,0.92,0.94,0.96,0.98,0.9999]
# p_values = [0.99]
        
# if "nytimes" in dataPath:
#     gammas = compute_gamma_for_Gauss(p_values, sampled_data, landmarks)
# else:
#     gammas = compute_gamma_for_others(p_values, sampled_data, landmarks, query)

# gammas = compute_gamma_for_Gauss(p_values, sampled_data, landmarks)
gammas = compute_gamma_for_others(p_values, sampled_data, landmarks, query)

for i in range(len(p_values)):
    print(f"gamma for p={p_values[i]}: {gammas[i]}")

# validate(0.8, sampled_data, landmarks, query)
