/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#ifdef FINTEGER
#undef FINTEGER
#endif
#define FINTEGER int

#define sgemv_ sgemv_faiss
#define sgemm_ sgemm_faiss
#define dgemm_ dgemm_faiss
#define ssyrk_ ssyrk_faiss
#define ssyev_ ssyev_faiss
#define dsyev_ dsyev_faiss
#define sgesvd_ sgesvd_faiss
#define dgesvd_ dgesvd_faiss
#define sgeqrf_ sgeqrf_faiss
#define sorgqr_ sorgqr_faiss

#include "faiss/AutoTune.h"
#include "faiss/Clustering.h"
#include "faiss/IVFlib.h"
#include "faiss/Index.h"
#include "faiss/Index2Layer.h"
#include "faiss/IndexAdditiveQuantizer.h"
#include "faiss/IndexAdditiveQuantizerFastScan.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryFromFloat.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexBinaryHash.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexFastScan.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexFlatCodes.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFAdditiveQuantizer.h"
#include "faiss/IndexIVFAdditiveQuantizerFastScan.h"
#include "faiss/IndexIVFFastScan.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFIndependentQuantizer.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/IndexIVFPQR.h"
#include "faiss/IndexIVFSpectralHash.h"
#include "faiss/IndexLSH.h"
#include "faiss/IndexLattice.h"
#include "faiss/IndexNNDescent.h"
#include "faiss/IndexNSG.h"
#include "faiss/IndexNeuralNetCodec.h"
#include "faiss/IndexPQ.h"
#include "faiss/IndexPQFastScan.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/IndexRefine.h"
#include "faiss/IndexReplicas.h"
#include "faiss/IndexRowwiseMinMax.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/IndexShards.h"
#include "faiss/IndexShardsIVF.h"
#include "faiss/MatrixStats.h"
#include "faiss/MetaIndexes.h"
#include "faiss/MetricType.h"
#include "faiss/VectorTransform.h"
#include "faiss/clone_index.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"

#undef sgemv_
#undef sgemm_
#undef dgemm_
#undef ssyrk_
#undef ssyev_
#undef dsyev_
#undef sgesvd_
#undef dgesvd_
#undef sgeqrf_
#undef sorgqr_