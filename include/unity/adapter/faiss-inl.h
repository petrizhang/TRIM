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

#include "./faiss-forwards.h"

#define Run_get_distance_computer Run_get_distance_computer_extra
#include "faiss/utils/extra_distances.cpp"
#undef Run_get_distance_computer

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

/************************************************************
 * header files
 **************************************************************/
#include "faiss/IndexIDMap.h"

/************************************************************
 * source files
 **************************************************************/
#include "./fake_omp.h"
#include "faiss/Clustering.cpp"
#include "faiss/Index.cpp"
#include "faiss/IndexBinary.cpp"
#include "faiss/IndexFlat.cpp"
#include "faiss/IndexFlatCodes.cpp"
#include "faiss/IndexIDMap.cpp"
#include "faiss/IndexIVF.cpp"
#include "faiss/IndexIVFFlat.cpp"
#include "faiss/IndexIVFPQ.cpp"
#include "faiss/IndexIVFPQR.cpp"
#include "faiss/IndexPQ.cpp"
#include "faiss/IndexPreTransform.cpp"
#include "faiss/IndexRefine.cpp"
#include "faiss/IndexScalarQuantizer.cpp"
#include "faiss/VectorTransform.cpp"
#include "faiss/impl/AuxIndexStructures.cpp"
#include "faiss/impl/CodePacker.cpp"
#include "faiss/impl/FaissException.cpp"
#include "faiss/impl/IDSelector.cpp"
#include "faiss/impl/PolysemousTraining.cpp"
#include "faiss/impl/ProductQuantizer.cpp"
#include "faiss/impl/ScalarQuantizer.cpp"
#include "faiss/impl/io.cpp"
#include "faiss/impl/kmeans1d.cpp"
#include "faiss/invlists/BlockInvertedLists.cpp"
#include "faiss/invlists/DirectMap.cpp"
#include "faiss/invlists/InvertedLists.cpp"
#include "faiss/invlists/InvertedListsIOHook.cpp"
#include "faiss/invlists/OnDiskInvertedLists.cpp"
#include "faiss/utils/Heap.cpp"
#include "faiss/utils/distances.cpp"
#include "faiss/utils/distances_fused/distances_fused.cpp"
#include "faiss/utils/distances_fused/simdlib_based.cpp"
#include "faiss/utils/distances_simd.cpp"
#include "faiss/utils/hamming.cpp"
#include "faiss/utils/partitioning.cpp"
#include "faiss/utils/random.cpp"
#include "faiss/utils/sorting.cpp"
#include "faiss/utils/utils.cpp"

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