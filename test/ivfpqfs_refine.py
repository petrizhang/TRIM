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

# index = faiss.index_factory(d, f"OPQ{m},PQ{m}x{nbits}")
index = faiss.index_factory(d, f"IVF{16},PQ{d//2}x{4}fs,RFlat")

# Train the index (if necessary)
index.train(vectors)

# Add vectors to the index
index.add(vectors)

D, I = index.search(vectors[0].reshape(1, d), k=10)
print(D)
print(I)
