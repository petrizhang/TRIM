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

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>

#include "faiss/IndexFlat.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexIVFPQR.h"
#include "faiss/IndexPQ.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/IndexRefine.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/VectorTransform.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/index_read_utils.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/index_io.h"
#include "faiss/invlists/InvertedListsIOHook.h"
#include "faiss/utils/hamming.h"

namespace faiss {

Index* unity_read_index(const char* fname, int io_flags = 0);
Index* unity_read_index(FILE* f, int io_flags = 0);
Index* unity_read_index(IOReader* reader, int io_flags = 0);
VectorTransform* unity_read_VectorTransform(const char* fname);
VectorTransform* unity_read_VectorTransform(IOReader* f);
ProductQuantizer* unity_read_ProductQuantizer(const char* fname);
ProductQuantizer* unity_read_ProductQuantizer(IOReader* reader);
InvertedLists* unity_read_InvertedLists(IOReader* reader, int io_flags = 0);
void unity_read_index_header(Index* idx, IOReader* f);
void unity_read_direct_map(DirectMap* dm, IOReader* f);
void unity_read_ivf_header(IndexIVF* ivf, IOReader* f,
                           std::vector<std::vector<idx_t>>* ids = nullptr);
void unity_read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags);
ArrayInvertedLists* set_array_invlist(IndexIVF* ivf, std::vector<std::vector<idx_t>>& ids);
void unity_read_ProductQuantizer(ProductQuantizer* pq, IOReader* f);
void unity_read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f);

/*************************************************************
 * Read
 **************************************************************/

void unity_read_index_header(Index* idx, IOReader* f) {
  READ1(idx->d);
  READ1(idx->ntotal);
  idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  READ1(idx->is_trained);
  READ1(idx->metric_type);
  if (idx->metric_type > 1) {
    READ1(idx->metric_arg);
  }
  idx->verbose = false;
}

VectorTransform* unity_read_VectorTransform(IOReader* f) {
  uint32_t h;
  READ1(h);
  VectorTransform* vt = nullptr;

  if (h == fourcc("rrot") || h == fourcc("PCAm") || h == fourcc("LTra") || h == fourcc("PcAm") ||
      h == fourcc("Viqm") || h == fourcc("Pcam")) {
    LinearTransform* lt = nullptr;
    if (h == fourcc("rrot")) {
      lt = new RandomRotationMatrix();
    } else if (h == fourcc("PCAm") || h == fourcc("PcAm") || h == fourcc("Pcam")) {
      PCAMatrix* pca = new PCAMatrix();
      READ1(pca->eigen_power);
      if (h == fourcc("Pcam")) {
        READ1(pca->epsilon);
      }
      READ1(pca->random_rotation);
      if (h != fourcc("PCAm")) {
        READ1(pca->balanced_bins);
      }
      READVECTOR(pca->mean);
      READVECTOR(pca->eigenvalues);
      READVECTOR(pca->PCAMat);
      lt = pca;
    } else if (h == fourcc("Viqm")) {
      ITQMatrix* itqm = new ITQMatrix();
      READ1(itqm->max_iter);
      READ1(itqm->seed);
      lt = itqm;
    } else if (h == fourcc("LTra")) {
      lt = new LinearTransform();
    }
    READ1(lt->have_bias);
    READVECTOR(lt->A);
    READVECTOR(lt->b);
    FAISS_THROW_IF_NOT(lt->A.size() >= lt->d_in * lt->d_out);
    FAISS_THROW_IF_NOT(!lt->have_bias || lt->b.size() >= lt->d_out);
    lt->set_is_orthonormal();
    vt = lt;
  } else {
    FAISS_THROW_FMT("fourcc %ud (\"%s\") not recognized in %s", h, fourcc_inv_printable(h).c_str(),
                    f->name.c_str());
  }
  READ1(vt->d_in);
  READ1(vt->d_out);
  READ1(vt->is_trained);
  return vt;
}

static void unity_read_ArrayInvertedLists_sizes(IOReader* f, std::vector<size_t>& sizes) {
  uint32_t list_type;
  READ1(list_type);
  if (list_type == fourcc("full")) {
    size_t os = sizes.size();
    READVECTOR(sizes);
    FAISS_THROW_IF_NOT(os == sizes.size());
  } else if (list_type == fourcc("sprs")) {
    std::vector<size_t> idsizes;
    READVECTOR(idsizes);
    for (size_t j = 0; j < idsizes.size(); j += 2) {
      FAISS_THROW_IF_NOT(idsizes[j] < sizes.size());
      sizes[idsizes[j]] = idsizes[j + 1];
    }
  } else {
    FAISS_THROW_FMT("list_type %ud (\"%s\") not recognized", list_type,
                    fourcc_inv_printable(list_type).c_str());
  }
}

InvertedLists* unity_read_InvertedLists(IOReader* f, int io_flags) {
  uint32_t h;
  READ1(h);
  if (h == fourcc("il00")) {
    fprintf(stderr,
            "unity_read_InvertedLists:"
            " WARN! inverted lists not stored with IVF object\n");
    return nullptr;
  } else if (h == fourcc("ilar") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
    auto ails = new ArrayInvertedLists(0, 0);
    READ1(ails->nlist);
    READ1(ails->code_size);
    ails->ids.resize(ails->nlist);
    ails->codes.resize(ails->nlist);
    std::vector<size_t> sizes(ails->nlist);
    unity_read_ArrayInvertedLists_sizes(f, sizes);
    for (size_t i = 0; i < ails->nlist; i++) {
      ails->ids[i].resize(sizes[i]);
      ails->codes[i].resize(sizes[i] * ails->code_size);
    }
    for (size_t i = 0; i < ails->nlist; i++) {
      size_t n = ails->ids[i].size();
      if (n > 0) {
        READANDCHECK(ails->codes[i].data(), n * ails->code_size);
        READANDCHECK(ails->ids[i].data(), n);
      }
    }
    return ails;

  } else if (h == fourcc("ilar") && (io_flags & IO_FLAG_SKIP_IVF_DATA)) {
    // code is always ilxx where xx is specific to the type of invlists we
    // want so we get the 16 high bits from the io_flag and the 16 low bits
    // as "il"
    int h2 = (io_flags & 0xffff0000) | (fourcc("il__") & 0x0000ffff);
    size_t nlist, code_size;
    READ1(nlist);
    READ1(code_size);
    std::vector<size_t> sizes(nlist);
    unity_read_ArrayInvertedLists_sizes(f, sizes);
    return InvertedListsIOHook::lookup(h2)->read_ArrayInvertedLists(f, io_flags, nlist, code_size,
                                                                    sizes);
  } else {
    return InvertedListsIOHook::lookup(h)->read(f, io_flags);
  }
}

void unity_read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags) {
  InvertedLists* ils = unity_read_InvertedLists(f, io_flags);
  if (ils) {
    FAISS_THROW_IF_NOT(ils->nlist == ivf->nlist);
    FAISS_THROW_IF_NOT(ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
                       ils->code_size == ivf->code_size);
  }
  ivf->invlists = ils;
  ivf->own_invlists = true;
}

void unity_read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f) {
  READ1(ivsc->qtype);
  READ1(ivsc->rangestat);
  READ1(ivsc->rangestat_arg);
  READ1(ivsc->d);
  READ1(ivsc->code_size);
  READVECTOR(ivsc->trained);
  ivsc->set_derived_sizes();
}

void unity_read_ProductQuantizer(ProductQuantizer* pq, IOReader* f) {
  READ1(pq->d);
  READ1(pq->M);
  READ1(pq->nbits);
  pq->set_derived_values();
  READVECTOR(pq->centroids);
}

ProductQuantizer* unity_read_ProductQuantizer(IOReader* reader) {
  ProductQuantizer* pq = new ProductQuantizer();
  std::unique_ptr<ProductQuantizer> del(pq);

  unity_read_ProductQuantizer(pq, reader);
  del.release();
  return pq;
}

ProductQuantizer* unity_read_ProductQuantizer(const char* fname) {
  FileIOReader reader(fname);
  return unity_read_ProductQuantizer(&reader);
}

void unity_read_direct_map(DirectMap* dm, IOReader* f) {
  char maintain_direct_map;
  READ1(maintain_direct_map);
  dm->type = (DirectMap::Type)maintain_direct_map;
  READVECTOR(dm->array);
  if (dm->type == DirectMap::Hashtable) {
    std::vector<std::pair<idx_t, idx_t>> v;
    READVECTOR(v);
    std::unordered_map<idx_t, idx_t>& map = dm->hashtable;
    map.reserve(v.size());
    for (auto it : v) {
      map[it.first] = it.second;
    }
  }
}

void unity_read_ivf_header(IndexIVF* ivf, IOReader* f, std::vector<std::vector<idx_t>>* ids) {
  unity_read_index_header(ivf, f);
  READ1(ivf->nlist);
  READ1(ivf->nprobe);
  ivf->quantizer = unity_read_index(f);
  ivf->own_fields = true;
  if (ids) {  // used in legacy "Iv" formats
    ids->resize(ivf->nlist);
    for (size_t i = 0; i < ivf->nlist; i++) READVECTOR((*ids)[i]);
  }
  unity_read_direct_map(&ivf->direct_map, f);
}

// used for legacy formats
ArrayInvertedLists* unity_set_array_invlist(IndexIVF* ivf, std::vector<std::vector<idx_t>>& ids) {
  ArrayInvertedLists* ail = new ArrayInvertedLists(ivf->nlist, ivf->code_size);
  std::swap(ail->ids, ids);
  ivf->invlists = ail;
  ivf->own_invlists = true;
  return ail;
}

static IndexIVFPQ* unity_read_ivfpq(IOReader* f, uint32_t h, int io_flags) {
  bool legacy = h == fourcc("IvQR") || h == fourcc("IvPQ");

  IndexIVFPQR* ivfpqr = h == fourcc("IvQR") || h == fourcc("IwQR") ? new IndexIVFPQR() : nullptr;
  IndexIVFPQ* ivpq = ivfpqr ? ivfpqr : new IndexIVFPQ();

  std::vector<std::vector<idx_t>> ids;
  unity_read_ivf_header(ivpq, f, legacy ? &ids : nullptr);
  READ1(ivpq->by_residual);
  READ1(ivpq->code_size);
  unity_read_ProductQuantizer(&ivpq->pq, f);

  if (legacy) {
    ArrayInvertedLists* ail = unity_set_array_invlist(ivpq, ids);
    for (size_t i = 0; i < ail->nlist; i++) READVECTOR(ail->codes[i]);
  } else {
    unity_read_InvertedLists(ivpq, f, io_flags);
  }

  if (ivpq->is_trained) {
    // precomputed table not stored. It is cheaper to recompute it.
    // precompute_table() may be disabled with a flag.
    ivpq->use_precomputed_table = 0;
    if (ivpq->by_residual) {
      if ((io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) == 0) {
        ivpq->precompute_table();
      }
    }
    if (ivfpqr) {
      unity_read_ProductQuantizer(&ivfpqr->refine_pq, f);
      READVECTOR(ivfpqr->refine_codes);
      READ1(ivfpqr->k_factor);
    }
  }
  return ivpq;
}

int unity_read_old_fmt_hack = 0;

Index* unity_read_index(IOReader* f, int io_flags) {
  Index* idx = nullptr;
  uint32_t h;
  READ1(h);
  if (h == fourcc("null")) {
    // denotes a missing index, useful for some cases
    return nullptr;
  } else if (h == fourcc("IxFI") || h == fourcc("IxF2") || h == fourcc("IxFl")) {
    IndexFlat* idxf;
    if (h == fourcc("IxFI")) {
      idxf = new IndexFlatIP();
    } else if (h == fourcc("IxF2")) {
      idxf = new IndexFlatL2();
    } else {
      idxf = new IndexFlat();
    }
    unity_read_index_header(idxf, f);
    idxf->code_size = idxf->d * sizeof(float);
    READXBVECTOR(idxf->codes);
    FAISS_THROW_IF_NOT(idxf->codes.size() == idxf->ntotal * idxf->code_size);
    // leak!
    idx = idxf;
  } else if (h == fourcc("IxPQ") || h == fourcc("IxPo") || h == fourcc("IxPq")) {
    // IxPQ and IxPo were merged into the same IndexPQ object
    IndexPQ* idxp = new IndexPQ();
    unity_read_index_header(idxp, f);
    unity_read_ProductQuantizer(&idxp->pq, f);
    idxp->code_size = idxp->pq.code_size;
    READVECTOR(idxp->codes);
    if (h == fourcc("IxPo") || h == fourcc("IxPq")) {
      READ1(idxp->search_type);
      READ1(idxp->encode_signs);
      READ1(idxp->polysemous_ht);
    }
    // Old versions of PQ all had metric_type set to INNER_PRODUCT
    // when they were in fact using L2. Therefore, we force metric type
    // to L2 when the old format is detected
    if (h == fourcc("IxPQ") || h == fourcc("IxPo")) {
      idxp->metric_type = METRIC_L2;
    }
    idx = idxp;
  } else if (h == fourcc("IvFl") || h == fourcc("IvFL")) {  // legacy
    IndexIVFFlat* ivfl = new IndexIVFFlat();
    std::vector<std::vector<idx_t>> ids;
    unity_read_ivf_header(ivfl, f, &ids);
    ivfl->code_size = ivfl->d * sizeof(float);
    ArrayInvertedLists* ail = unity_set_array_invlist(ivfl, ids);

    if (h == fourcc("IvFL")) {
      for (size_t i = 0; i < ivfl->nlist; i++) {
        READVECTOR(ail->codes[i]);
      }
    } else {  // old format
      for (size_t i = 0; i < ivfl->nlist; i++) {
        std::vector<float> vec;
        READVECTOR(vec);
        ail->codes[i].resize(vec.size() * sizeof(float));
        memcpy(ail->codes[i].data(), vec.data(), ail->codes[i].size());
      }
    }
    idx = ivfl;
  } else if (h == fourcc("IwFl")) {
    IndexIVFFlat* ivfl = new IndexIVFFlat();
    unity_read_ivf_header(ivfl, f);
    ivfl->code_size = ivfl->d * sizeof(float);
    unity_read_InvertedLists(ivfl, f, io_flags);
    idx = ivfl;
  } else if (h == fourcc("IvPQ") || h == fourcc("IvQR") || h == fourcc("IwPQ") ||
             h == fourcc("IwQR")) {
    idx = unity_read_ivfpq(f, h, io_flags);
  } else if (h == fourcc("IxPT")) {
    IndexPreTransform* ixpt = new IndexPreTransform();
    ixpt->own_fields = true;
    unity_read_index_header(ixpt, f);
    int nt;
    if (unity_read_old_fmt_hack == 2) {
      nt = 1;
    } else {
      READ1(nt);
    }
    for (int i = 0; i < nt; i++) {
      ixpt->chain.push_back(unity_read_VectorTransform(f));
    }
    ixpt->index = unity_read_index(f, io_flags);
    idx = ixpt;
  } else if (h == fourcc("IxRF")) {
    IndexRefine* idxrf = new IndexRefine();
    unity_read_index_header(idxrf, f);
    idxrf->base_index = unity_read_index(f, io_flags);
    idxrf->refine_index = unity_read_index(f, io_flags);
    READ1(idxrf->k_factor);
    if (dynamic_cast<IndexFlat*>(idxrf->refine_index)) {
      // then make a RefineFlat with it
      IndexRefine* idxrf_old = idxrf;
      idxrf = new IndexRefineFlat();
      *idxrf = *idxrf_old;
      delete idxrf_old;
    }
    idxrf->own_fields = true;
    idxrf->own_refine_index = true;
    idx = idxrf;
  } else if (h == fourcc("IxMp") || h == fourcc("IxM2")) {
    bool is_map2 = h == fourcc("IxM2");
    IndexIDMap* idxmap = is_map2 ? new IndexIDMap2() : new IndexIDMap();
    unity_read_index_header(idxmap, f);
    idxmap->index = unity_read_index(f, io_flags);
    idxmap->own_fields = true;
    READVECTOR(idxmap->id_map);
    if (is_map2) {
      static_cast<IndexIDMap2*>(idxmap)->construct_rev_map();
    }
    idx = idxmap;
  } else {
    FAISS_THROW_FMT("Index type 0x%08x (\"%s\") not recognized", h,
                    fourcc_inv_printable(h).c_str());
    idx = nullptr;
  }
  return idx;
}

Index* unity_read_index(FILE* f, int io_flags) {
  FileIOReader reader(f);
  return unity_read_index(&reader, io_flags);
}

Index* unity_read_index(const char* fname, int io_flags) {
  FileIOReader reader(fname);
  Index* idx = unity_read_index(&reader, io_flags);
  return idx;
}

VectorTransform* unity_read_VectorTransform(const char* fname) {
  FileIOReader reader(fname);
  VectorTransform* vt = unity_read_VectorTransform(&reader);
  return vt;
}

}  // namespace faiss
