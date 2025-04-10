#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include "faiss/IndexIVFPQ.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "trim/detail/io/read_faiss.h"

namespace faiss {

struct tIVFPQ: IndexIVFPQ{
    
    std::vector<float> recons_errors;

    float* data{nullptr};

    // load IVFPQ
    tIVFPQ(const char* index_path){
        // load IVFPQ
        FileIOReader reader(index_path);
        IOReader* f = &reader;
        uint32_t h;
        int io_flags = 0;
        READ1(h);
        if (h == fourcc("IvPQ") || h == fourcc("IvQR") || h == fourcc("IwPQ") || h == fourcc("IwQR")) {
            bool legacy = h == fourcc("IvQR") || h == fourcc("IvPQ");

            // IndexIVFPQR* ivfpqr = h == fourcc("IvQR") || h == fourcc("IwQR") ? new IndexIVFPQR() : nullptr;
            // IndexIVFPQ* ivpq = ivfpqr ? ivfpqr : new IndexIVFPQ();
            
            std::vector<std::vector<idx_t>> ids;
            trim_read_ivf_header(this, f, legacy ? &ids : nullptr);
            READ1(by_residual);
            READ1(code_size);
            trim_read_ProductQuantizer(&pq, f);
            
            if (legacy) {
                ArrayInvertedLists* ail = trim_set_array_invlist(this, ids);
                for (size_t i = 0; i < ail->nlist; i++) READVECTOR(ail->codes[i]);
            } else {
                trim_read_InvertedLists(this, f, io_flags);
            }
            
            if (is_trained) {
                // precomputed table not stored. It is cheaper to recompute it.
                // precompute_table() may be disabled with a flag.
                use_precomputed_table = 0;
                if (by_residual) {
                    if ((io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) == 0) {
                        precompute_table();
                    }
                }
                // if (ivfpqr) {
                // trim_read_ProductQuantizer(&ivfpqr->refine_pq, f);
                // READVECTOR(ivfpqr->refine_codes);
                // READ1(ivfpqr->k_factor);
                // }
            }
        }else throw std::invalid_argument("Wrong index type");
        
        this->metric_type = METRIC_L2;
        // this->parallel_mode = PARALLEL_MODE_NO_HEAP_INIT;
        recons_errors.resize(ntotal, 0.0);
    }
 
    ~tIVFPQ() {}

    void set_data(float* data) {
        this->data = data;
    }

    const float* get_data(unsigned i) const {
        
        if (i < 0 || i >= ntotal) {
            throw std::out_of_range("Vector index out of range: " + std::to_string(i));
        }
        // float* vec = new float[ivfpq->d];
        // std::memcpy(vec, data + i * ivfpq->d, ivfpq->d * sizeof(float));

        return data + i * d;
    }

    void compute_recons_errors() {
        
        if (!invlists) {
            std::cerr << "Error: invlists is null." << std::endl;
            return;
        }

        auto array_invlists = dynamic_cast<ArrayInvertedLists*>(invlists);

        for (int list_no = 0; list_no < nlist; list_no++) {
            
            size_t list_size = invlists->list_size(list_no);
            
            if (list_size == 0) {
                continue;
            }
    
            const idx_t* ids = array_invlists->get_ids(list_no);

            for(size_t i = 0; i < list_size; i++){
                
                idx_t id = ids[i];
                
                float* x_approx = new float[this->d];
                reconstruct_from_offset(list_no, i, x_approx);

                float error = fvec_L2sqr(x_approx, id);
                recons_errors[id] = std::sqrt(error);

            }
        }

    }

#ifdef USE_AVX
    float fvec_L2sqr(const float* x, idx_t id) const {
    
        size_t d = this->d;
        const float* y = get_data(id);
    
        __m256 sum = _mm256_setzero_ps();
        size_t i;

        for (i = 0; i < d - 7; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 y_vec = _mm256_loadu_ps(y + i);
            __m256 diff = _mm256_sub_ps(x_vec, y_vec);
            __m256 sq = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, sq);
        }

        float res[8];
        _mm256_storeu_ps(res, sum);
        float result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];

        for (; i < d; i++) {
            const float tmp = x[i] - y[i];
            result += tmp * tmp;
        }

        return result;
         
    }
#else
    float fvec_L2sqr(const float* x, idx_t id) const {
        const float* y = get_data(id);

        size_t i;
        float res = 0;
        FAISS_PRAGMA_IMPRECISE_LOOP
        for (i = 0; i < this->d; i++) {
            const float tmp = x[i] - y[i];
            res += tmp * tmp;
        }
        
        return res;
    }
#endif

    
};

} // namespace faiss