import numpy as np
import faiss

# Set the random seed for reproducible results
np.random.seed(0)

# Generate 1000 vectors with 256 dimensions from a standard normal distribution
vectors = np.random.randn(1000, 256).astype('float32')

# Print the shape of the generated vectors
print(f"Generated {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")

# Define the parameters for the PQ index
d = vectors.shape[1]  # Dimensionality of the vectors
m = 16  # Code length per subquantizer
nbits = 8  # Number of subquantizers

# Create a PQ index using FAISS's index factory
index = faiss.IndexPQ(d, m, nbits)  # 8 is the length of the codes

# Train the index (if necessary)
index.train(vectors)

# Add vectors to the index
index.add(vectors)

# Save the index to a file
faiss.write_index(index, "index_pq.bin")

# Print information about the saved index
print("Index saved to 'index_pq.bin'")