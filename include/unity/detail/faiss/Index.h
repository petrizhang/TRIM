/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <unity/detail/faiss/MetricType.h>

namespace faiss {

enum IndexType { INDEX_PQ = 1, INDEX_IVFPQ_FLAT = 0 };

/** Abstract structure for an index, supports adding vectors and searching
 * them.
 *
 * All vectors provided at add or search time are 32-bit float arrays,
 * although the internal representation may vary.
 */
struct Index {
  IndexType index_type;  ///< faiss index type
  int d;                 ///< vector dimension
  idx_t ntotal;          ///< total nb of indexed vectors
  bool verbose;          ///< verbosity level

  /// set if the Index does not require training, or if training is
  /// done already
  bool is_trained;

  /// type of metric this index uses for search
  MetricType metric_type;
  float metric_arg;  ///< argument of the metric type
};

}  // namespace faiss
