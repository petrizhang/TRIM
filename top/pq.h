#include <cstdint>
struct PQ {
  size_t d;         ///< size of the input vectors
  size_t code_size; ///< bytes per indexed vector

  int d;        ///< vector dimension
  idx_t ntotal; ///< total nb of indexed vectors

  /// encoded dataset, size ntotal * code_size
  std::vector<uint8_t> codes;

      size_t M;     ///< number of subquantizers
    size_t nbits; ///< number of bits per quantization index

    // values derived from the above
    size_t dsub;  ///< dimensionality of each subvector
    size_t ksub;  ///< number of centroids for each subquantizer
    bool verbose; ///< verbose during training?

    /// initialization
    enum train_type_t {
        Train_default,
        Train_hot_start,     ///< the centroids are already initialized
        Train_shared,        ///< share dictionary across PQ segments
        Train_hypercube,     ///< initialize centroids with nbits-D hypercube
        Train_hypercube_pca, ///< initialize centroids with nbits-D hypercube
    };
    train_type_t train_type;

    ClusteringParameters cp; ///< parameters used during clustering

    /// if non-NULL, use this index for assignment (should be of size
    /// d / M)
    Index* assign_index;

    /// Centroid table, size M * ksub * dsub.
    /// Layout: (M, ksub, dsub)
    std::vector<float> centroids;

    /// Squared lengths of centroids, size M * ksub
    /// Layout: (M, ksub)
    std::vector<float> centroids_sq_lengths;

};
