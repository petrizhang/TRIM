import faiss
import hnswlib
import numpy as np

# Set the random seed for reproducible results
np.random.seed(42)

# Generate 1000 vectors with 256 dimensions from a standard normal distribution
vectors = np.random.randn(1000, 256).astype('float32')

# Print the shape of the generated vectors
print(f"Generated {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")

# Define the parameters for the PQ index
d = vectors.shape[1]  # Dimensionality of the vectors
m = 16  # Code length per subquantizer
nbits = 8  # Number of subquantizers

use_fast_scan = True
# Create a PQ index using FAISS's index factory
if use_fast_scan:
    # index = faiss.index_factory(d, f"OPQ{m},PQ{m}x{nbits}")
    index = faiss.index_factory(d, f"IVF{16},PQ{d//2}x{4}fs")
else:
    index = faiss.IndexPQFastScan(d, m, nbits)  # 8 is the length of the codes

# Train the index (if necessary)
index.train(vectors)

# Add vectors to the index
index.add(vectors)

# Save the index to a file
index_file = "index_ivfpqfs.bin" if use_fast_scan else "index_pq.bin"
faiss.write_index(index, index_file)

# Print information about the saved index
print(f"Index saved to '{index_file}'")


# 加载索引
index = hnswlib.Index(space='l2', dim=256)

# 初始化索引，设置最大元素数量、ef_construction和M
index.init_index(max_elements=1000, ef_construction=200, M=16)

# 将数据添加到索引中
index.add_items(vectors)
index.save_index("hnswlib.bin")
