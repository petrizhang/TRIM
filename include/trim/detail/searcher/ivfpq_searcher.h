#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <type_traits>
// #include <H5Cpp.h>
#include <iomanip>

#include "trim/common/common.h"
#include "trim/common/prefetch.h"
#include "trim/common/searcher.h"
#include "trim/common/setter_proxy.h"
#include "trim/detail/index/tIVFPQ.h"
#include <faiss/utils/distances.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/code_distance/code_distance.h>
#include <faiss/impl/AuxIndexStructures.h>

// using namespace H5;
namespace trim {
namespace detail {
    struct IVFPQSearcher : SetterProxy<IVFPQSearcher>, ISearcher {  
        using dist_t = float;
        using idx_t = faiss::idx_t;
        using This = IVFPQSearcher;
        using Proxy = SetterProxy<This>;
        using RangeSearchResult = faiss::RangeSearchResult;

        // Parameters
        size_t _nprobe{30};
        float _k_factor{10};
        float _gamma{0.8};
        mutable float _pruning_ratio{0.0f};
        mutable float _actual_distance_computation{0.0f};
        mutable float _total_distance_computation{0.0f};
        bool _trim_opened{true};

        //Index
        faiss::tIVFPQ* _ivfpq{nullptr};   

        // data
        const float* _data{nullptr};

        U_FORBID_COPY_AND_ASSIGN(IVFPQSearcher);

        U_FORBID_MOVE(IVFPQSearcher);

        IVFPQSearcher() = delete;

        explicit IVFPQSearcher(const char* index_path):Proxy("IVFPQSearcher"){  
            _ivfpq = new faiss::tIVFPQ(index_path);
            _pruning_ratio = 0.0f;
            _actual_distance_computation = 0.0f;
            _total_distance_computation = 0.0f;
            
            Proxy::template bind<BOOL_TYPE>("trim_opened", &This::set_trim_opened);
            Proxy::template bind<DOUBLE_TYPE>("k_factor", &This::set_k_factor);
            Proxy::template bind<INTEGER_TYPE>("nprobe", &This::set_nprobe);
            Proxy::template bind<DOUBLE_TYPE>("gamma", &This::set_gamma);
        }

        ~IVFPQSearcher() override = default;

        void set_data(float* data) override {
            this->_data = data;
            _ivfpq->set_data(data);
            _ivfpq->compute_recons_errors();
        }

        void set(const std::string& key, const Object& value) override { Proxy::proxy_set(key, value); }

        void try_set(const std::string& key, const Object& value) override {Proxy::proxy_try_set(key, value);}

        const float* get_data(unsigned i) const override {return _ivfpq->get_data(i);}

        size_t num_data_points() const override {return _ivfpq == nullptr ? 0 : _ivfpq->ntotal;}

        void set_k_factor(float kf) { this->_k_factor = kf; }

        void set_nprobe(size_t nprobe) {this->_nprobe = nprobe;}

        void set_gamma(float gamma) {this->_gamma = gamma;}

        void set_trim_opened(bool flag) {this->_trim_opened = flag;}

        void optimize(int num_threads) override { T_THROW_MSG("not implemented error"); }

        IDCO* get_dco() const override { T_THROW_MSG("not implemented error"); }

        Dict get_profile() const override { T_THROW_MSG("not implemented error");}

        float get_pruning_ratio() const override { return _pruning_ratio; }

        float get_actual_distance_computation() const override { return _actual_distance_computation; }

        float get_total_distance_computation() const override { return _total_distance_computation; }
  
        void clear_pruning_ratio() const override { _pruning_ratio = 0.0; }

        void clear_num_distance_computation() const override { _actual_distance_computation = 0.0; _total_distance_computation = 0.0; }
        
        void ann_search(const float* q, int k, int* dst) const override {
           
            // omp_set_num_threads(1);
            
            if(_trim_opened)
                _ann_search_with_trim_2(q, k, dst);
            else
                _ann_search_original(q, k, dst);
        }

        void _ann_search_original(const float* q, int k, int* dst) const{
        
            FAISS_THROW_IF_NOT(k > 0);
    
            // const int D = _ivfpq->d;
            faiss::SearchParametersIVF params;
            params.nprobe = _nprobe;
            int can_size = std::max(k, (int)(k *_k_factor));
            float* approx_distances = new float[can_size];
            idx_t* ids = new idx_t[can_size];
            std::fill_n(ids, can_size, -1);  
            
            _ivfpq->search(1, q, can_size, approx_distances, ids, &params);

            // refine
            float max_dist = 0;
            std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>> results_queue;
            for(size_t i=0; i<can_size; i++){
                if (ids[i] >= _ivfpq->ntotal || ids[i] == -1) {
                    continue;
                }
                float dist = _ivfpq->fvec_L2sqr(q, ids[i]);
                
                if(results_queue.size() < k || dist < max_dist){
                    results_queue.emplace(dist, ids[i]);
                    if(results_queue.size() > k) results_queue.pop();
                    if(!results_queue.empty()) max_dist = results_queue.top().first;
                }
                
            }
    
            // std::cout<< "My answer:" << std::endl;
            int i = results_queue.size();
            while(!results_queue.empty()){
                idx_t id = results_queue.top().second;
                // std::cout<< "id:" << id << ", dist: " << results_queue.top().first << std::endl;
                dst[i - 1] = id;
                results_queue.pop();
                i--;
            }

            _actual_distance_computation += can_size;
            _total_distance_computation += can_size;
            _pruning_ratio += 0.0;

            // Free allocated memory
            delete[] approx_distances;
            delete[] ids;

        }

        void _ann_search_with_trim(const float* q, int k, int* dst) const{
        
            FAISS_THROW_IF_NOT(k > 0);

            // const int D = _ivfpq->d;
            const int code_size = _ivfpq->code_size;
            size_t M = _ivfpq->pq.M;
            size_t ksub = _ivfpq->pq.ksub;
            size_t nbits = _ivfpq->pq.nbits;
            int ef = (int) k * _k_factor;

            // Allocate temporary memory for distances and indices
            std::unique_ptr<idx_t[]> coarse_ids(new idx_t[_nprobe]);
            std::unique_ptr<float[]> coarse_dists(new float[_nprobe]);

            // Use the quantizer to find the nprobe nearest centroids to the query
            _ivfpq->quantizer->search(1, q, _nprobe, coarse_dists.get(), coarse_ids.get());
            _ivfpq->invlists->prefetch_lists(coarse_ids.get(), _nprobe);

            // // Iterate over each selected list (centroid)
            // std::unique_ptr<faiss::InvertedListScanner> scanner(_ivfpq->get_InvertedListScanner(false, nullptr));
            // // Compute the distance table
            // scanner->set_query(q);

            // Compute inner_prod_table
            float* sim_table = new float[M * ksub];
            float* sim_table_2 = new float[M * ksub];
            _ivfpq->pq.compute_inner_prod_table(q, sim_table_2);
            
            int actual_access_count = k;
            int total_access_count = ef;
           
            std::priority_queue<std::pair<float, faiss::idx_t>, std::vector<std::pair<float, faiss::idx_t>>> top_candidates; // k nodes wth smallest PQ distances
            std::priority_queue<std::pair<float, faiss::idx_t>, std::vector<std::pair<float, faiss::idx_t>>> candidates; // ef nodes wth smallest PQ distance
            
            for (idx_t list_no = 0; list_no < _nprobe; list_no++) {
                
                size_t list_size = _ivfpq->invlists->list_size(coarse_ids[list_no]);
                
                if (list_size == 0) {
                    continue;
                }

                // Calculate the distance table
                float dis0 = coarse_dists[list_no];
                idx_t coarse_id = coarse_ids[list_no];
                faiss::fvec_madd(
                    M * ksub,
                    _ivfpq->precomputed_table.data() + coarse_id * ksub * M,
                    -2.0,
                    sim_table_2,
                    sim_table);

                const uint8_t* codes = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_codes(coarse_id);
                const idx_t* ids = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_ids(coarse_id);

                // Process codes in batches of 8
                for (size_t i = 0; i < list_size; i += 8) {
                    size_t batch_size = std::min<size_t>(8, list_size - i);

                    if (batch_size == 8) {
                        // Process full batch of 8
                        Dist8 dists;
                        _relaxed_lowerbound8(codes + i * code_size, ids + i, sim_table, dis0, dists);

                        for (size_t j = 0; j < 8; ++j) {
                            float lowerbound = dists[j];
                            faiss::idx_t id = ids[i + j];

                            if (top_candidates.size() < k || lowerbound < top_candidates.top().first) {
                                top_candidates.emplace(lowerbound, id);
                                if (top_candidates.size() > k) top_candidates.pop();
                            }

                            if (candidates.size() < ef || lowerbound < candidates.top().first) {
                                candidates.emplace(lowerbound, id);
                                if (candidates.size() > ef) candidates.pop();
                            }
                        }
                    } else {
                        // Process remaining elements individually
                        for (size_t j = 0; j < batch_size; ++j) {
                            const uint8_t* code = codes + (i + j) * code_size;
                            faiss::idx_t id = ids[i + j];

                            float dist_lq = dis0 + faiss::distance_single_code<faiss::PQDecoder8>(M, nbits, sim_table, code);
                            float dist_lx = _ivfpq->recons_errors[id];
                            float lowerbound = relaxed_lowerbound(std::sqrt(dist_lq), dist_lx);

                            if (top_candidates.size() < k || lowerbound < top_candidates.top().first) {
                                top_candidates.emplace(lowerbound, id);
                                if (top_candidates.size() > k) top_candidates.pop();
                            }

                            if (candidates.size() < ef || lowerbound < candidates.top().first) {
                                candidates.emplace(lowerbound, id);
                                if (candidates.size() > ef) candidates.pop();
                            }
                        }
                    }
                }
            }  
            
             // Refine the top-k candidates using exact distance
             std::priority_queue<std::pair<float, faiss::idx_t>, std::vector<std::pair<float, faiss::idx_t>>> results_queue;

             while(!top_candidates.empty()){
                idx_t id = top_candidates.top().second;
                float dist = _ivfpq->fvec_L2sqr(q, id);
                results_queue.emplace(dist, id);
                top_candidates.pop();
             }

             while (candidates.size() > k)
             {
                float lowerbound = candidates.top().first;

                if(lowerbound < results_queue.top().first){
                    actual_access_count++;
                    idx_t id = candidates.top().second;
                    float dist = _ivfpq->fvec_L2sqr(q, id);
                    results_queue.emplace(dist, id);
                    results_queue.pop();
                }

                candidates.pop();
            }

            int i = results_queue.size();
            while(!results_queue.empty()){
                idx_t id = results_queue.top().second;
                dst[i - 1] = id;
                results_queue.pop();
                i--;
            }

            _actual_distance_computation += actual_access_count;
            _total_distance_computation += total_access_count;
            _pruning_ratio += (1- 1.0*actual_access_count/total_access_count);
            
        }

        void _ann_search_with_trim_2(const float* q, int k, int* dst) const{
        
            FAISS_THROW_IF_NOT(k > 0);

            // const int D = _ivfpq->d;
            const int code_size = _ivfpq->code_size;
            size_t M = _ivfpq->pq.M;
            size_t ksub = _ivfpq->pq.ksub;
            size_t nbits = _ivfpq->pq.nbits;
            // int ef = (int) k * _k_factor;

            // Allocate temporary memory for distances and indices
            std::unique_ptr<idx_t[]> coarse_ids(new idx_t[_nprobe]);
            std::unique_ptr<float[]> coarse_dists(new float[_nprobe]);

            // Use the quantizer to find the nprobe nearest centroids to the query
            _ivfpq->quantizer->search(1, q, _nprobe, coarse_dists.get(), coarse_ids.get());
            _ivfpq->invlists->prefetch_lists(coarse_ids.get(), _nprobe);

            // Compute inner_prod_table
            float* sim_table = new float[M * ksub];
            float* sim_table_2 = new float[M * ksub];
            _ivfpq->pq.compute_inner_prod_table(q, sim_table_2);
            
            int actual_access_count = 0;
            int total_access_count = 0;
           
            std::priority_queue<std::pair<float, faiss::idx_t>, std::vector<std::pair<float, faiss::idx_t>>> results_queue;
            
            for (idx_t list_no = 0; list_no < _nprobe; list_no++) {
                
                size_t list_size = _ivfpq->invlists->list_size(coarse_ids[list_no]);
                
                if (list_size == 0) {
                    continue;
                }

                total_access_count += list_size;

                // Calculate the distance table
                float dis0 = coarse_dists[list_no];
                idx_t coarse_id = coarse_ids[list_no];
                faiss::fvec_madd(
                    M * ksub,
                    _ivfpq->precomputed_table.data() + coarse_id * ksub * M,
                    -2.0,
                    sim_table_2,
                    sim_table);

                const uint8_t* codes = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_codes(coarse_id);
                const idx_t* ids = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_ids(coarse_id);

                // Process codes in batches of 8
                for (size_t i = 0; i < list_size; i += 8) {
                    size_t batch_size = std::min<size_t>(8, list_size - i);

                    if (batch_size == 8) {
                        // Process full batch of 8
                        Dist8 dists;
                        _relaxed_lowerbound8(codes + i * code_size, ids + i, sim_table, dis0, dists);

                        for (size_t j = 0; j < 8; ++j) {
                            float lowerbound = dists[j];
                            faiss::idx_t id = ids[i + j];

                            if (results_queue.size() < k || lowerbound < results_queue.top().first) {
                                actual_access_count++;
                                float dist = _ivfpq->fvec_L2sqr(q, id);
                                results_queue.emplace(dist, id);
                                if(results_queue.size() > k) 
                                    results_queue.pop();
                            }
                        }
                    } else {
                        // Process remaining elements individually
                        for (size_t j = 0; j < batch_size; ++j) {
                            const uint8_t* code = codes + (i + j) * code_size;
                            faiss::idx_t id = ids[i + j];

                            float dist_lq = dis0 + faiss::distance_single_code<faiss::PQDecoder8>(M, nbits, sim_table, code);
                            float dist_lx = _ivfpq->recons_errors[id];
                            float lowerbound = relaxed_lowerbound(std::sqrt(dist_lq), dist_lx);

                            if (results_queue.size() < k || lowerbound < results_queue.top().first) {
                                actual_access_count++;
                                float dist = _ivfpq->fvec_L2sqr(q, id);
                                results_queue.emplace(dist, id);
                                if(results_queue.size() > k) 
                                    results_queue.pop();
                            }
                        }
                    }
                }
            }  

            int i = results_queue.size();
            while(!results_queue.empty()){
                idx_t id = results_queue.top().second;
                dst[i - 1] = id;
                results_queue.pop();
                i--;
            }

            _actual_distance_computation += actual_access_count;
            _total_distance_computation += total_access_count;
            _pruning_ratio += (1- 1.0*actual_access_count/total_access_count);
            
        }

        void range_search(const float* q, float radius, std::vector<int> &result) const override {
            
            // omp_set_num_threads(1);
            
            if(_trim_opened)
                _range_search_with_trim8_2(q, radius, result);
            else
                _range_search_original(q, radius, result);
        }

        void _range_search_original(const float* q, float radius, std::vector<int> &result) const{
        
            FAISS_THROW_IF_NOT(radius >= 0);
    
            // const int D = _ivfpq->d;
            faiss::SearchParametersIVF params;
            params.nprobe = _nprobe;

            RangeSearchResult result_set(1);
            _ivfpq->range_search(1, q, radius*_k_factor, &result_set, &params);

            // refine
            float radius2 = radius * radius;
            for(size_t i=result_set.lims[0]; i<result_set.lims[1]; i++){
                idx_t id = result_set.labels[i];
                float dist = _ivfpq->fvec_L2sqr(q, id);

                if(dist <= radius2){
                    result.push_back(id);
                }  
            }

            _actual_distance_computation += (result_set.lims[1] - result_set.lims[0]);
            _total_distance_computation += (result_set.lims[1] - result_set.lims[0]);
            _pruning_ratio += 0.0;
        }

        // void _range_search_with_trim(const float* q, float radius, std::vector<int> &result) const{
        
        //     FAISS_THROW_IF_NOT(radius >= 0);

        //     // const int D = _ivfpq->d;
        //     const int code_size = _ivfpq->code_size;

        //     // Allocate temporary memory for distances and indices
        //     std::unique_ptr<idx_t[]> coarse_ids(new idx_t[_nprobe]);
        //     std::unique_ptr<float[]> coarse_dists(new float[_nprobe]);

        //     // Use the quantizer to find the nprobe nearest centroids to the query
        //     _ivfpq->quantizer->search(1, q, _nprobe, coarse_dists.get(), coarse_ids.get());
        //     _ivfpq->invlists->prefetch_lists(coarse_ids.get(), _nprobe);

        //     // Iterate over each selected list (centroid)
        //     std::unique_ptr<faiss::InvertedListScanner> scanner(_ivfpq->get_InvertedListScanner(false, nullptr));
        //     // Compute the distance table
        //     scanner->set_query(q);
            
        //     int total_access_count = 0;
        //     std::vector<int> top_candidates;
        //     float radius2 = radius * radius;
        //     for (idx_t list_no = 0; list_no < _nprobe; list_no++) {
                
        //         size_t list_size = _ivfpq->invlists->list_size(coarse_ids[list_no]);
                
        //         if (list_size == 0) {
        //             continue;
        //         }

        //         total_access_count += list_size;
        //         scanner->set_list(coarse_ids[list_no], coarse_dists[list_no]);

        //         const uint8_t* codes = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_codes(coarse_ids[list_no]);
        //         const idx_t* ids = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_ids(coarse_ids[list_no]);

        //         for (size_t i = 0; i < list_size; i++) {
        //             idx_t id = ids[i];
        //             const uint8_t* code = codes + i * code_size;

        //             float dist_lq = std::sqrt(scanner->distance_to_code(code));
        //             float dist_lx = _ivfpq->recons_errors[id];
        //             float lowerbound = relaxed_lowerbound(dist_lq, dist_lx);

        //             if(lowerbound <= radius2){
        //                top_candidates.push_back(id);
        //             }
        //         }
        //     }  
            
        //     int actual_access_count = top_candidates.size();
        //     for(size_t i=0; i<top_candidates.size(); i++){
        //         faiss::idx_t id = top_candidates[i];
        //         float dist = _ivfpq->fvec_L2sqr(q, id);

        //         if(dist <= radius2){
        //             result.push_back(id);
        //         }  
        //     }
            
        //     _actual_distance_computation += actual_access_count;
        //     _total_distance_computation += total_access_count;
        //     _pruning_ratio += (1- 1.0*actual_access_count/total_access_count);
             
        // }

        void _range_search_with_trim8(const float* q, float radius, std::vector<int> &result) const{
        
            FAISS_THROW_IF_NOT(radius >= 0);

            // const int D = _ivfpq->d;
            const int code_size = _ivfpq->code_size;
            size_t M = _ivfpq->pq.M;
            size_t ksub = _ivfpq->pq.ksub;
            size_t nbits = _ivfpq->pq.nbits;

            // Allocate temporary memory for distances and indices
            std::unique_ptr<idx_t[]> coarse_ids(new idx_t[_nprobe]);
            std::unique_ptr<float[]> coarse_dists(new float[_nprobe]);

            // Use the quantizer to find the nprobe nearest centroids to the query
            _ivfpq->quantizer->search(1, q, _nprobe, coarse_dists.get(), coarse_ids.get());
            _ivfpq->invlists->prefetch_lists(coarse_ids.get(), _nprobe);
            
            int total_access_count = 0;
            std::vector<int> top_candidates;
            float radius2 = radius * radius;

            for (idx_t list_no = 0; list_no < _nprobe; list_no++) {
                
                size_t list_size = _ivfpq->invlists->list_size(coarse_ids[list_no]);
                
                if (list_size == 0) {
                    continue;
                }

                total_access_count += list_size;

                // Calculate the distance table
                std::unique_ptr<float[]> dis_table(new float[M * ksub]);
                faiss::idx_t coarse_id = coarse_ids[list_no];
                std::vector<float> residual(_ivfpq->d);
                _ivfpq->quantizer->compute_residual(q, residual.data(), coarse_id);
                _ivfpq->pq.compute_distance_table(residual.data(), dis_table.get());

                const uint8_t* codes = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_codes(coarse_ids[list_no]);
                const idx_t* ids = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_ids(coarse_ids[list_no]);

                // Process codes in batches of 8
                for (size_t i = 0; i < list_size; i += 8) {
                    size_t batch_size = std::min<size_t>(8, list_size - i); // Handle the last batch if < 8

                    if (batch_size == 8) {
                        // std::cout << "batch 8"<< std::endl;
                        Dist8 dists;
                        const uint8_t* batch_codes = codes + i * code_size;
                        const idx_t* batch_ids = ids + i;

                        relaxed_lowerbound8(batch_codes, batch_ids, dis_table.get(), 0, dists);

                        for (size_t j = 0; j < 8; ++j) {
                            if (dists[j] <= radius2) {
                                top_candidates.push_back(batch_ids[j]);
                            }
                        }
                    } else {
                        for (size_t j = 0; j < batch_size; ++j) {
                            idx_t id = ids[i + j];
                            const uint8_t* code = codes + (i + j) * code_size;

                            float dist2_lq = faiss::distance_single_code<faiss::PQDecoder8>(M, nbits, dis_table.get(), code);
                            float dist_lq = std::sqrt(dist2_lq);
                            float dist_lx = _ivfpq->recons_errors[id];
                            float lowerbound = relaxed_lowerbound(dist_lq, dist_lx);

                            if (lowerbound <= radius2) {
                                top_candidates.push_back(id);
                            }
                        }
                    }
                }
            }  
            
            
            int actual_access_count = top_candidates.size();
            for(size_t i=0; i<top_candidates.size(); i++){
                faiss::idx_t id = top_candidates[i];
                float dist = _ivfpq->fvec_L2sqr(q, id);

                if(dist <= radius2){
                    result.push_back(id);
                }  
            }
            
            _actual_distance_computation += actual_access_count;
            _total_distance_computation += total_access_count;
            _pruning_ratio += (1- 1.0*actual_access_count/total_access_count);
             
        }

        void _range_search_with_trim8_2(const float* q, float radius, std::vector<int> &result) const{
        
            FAISS_THROW_IF_NOT(radius >= 0);

            // const int D = _ivfpq->d;
            const int code_size = _ivfpq->code_size;
            size_t M = _ivfpq->pq.M;
            size_t ksub = _ivfpq->pq.ksub;
            size_t nbits = _ivfpq->pq.nbits;

            // Allocate temporary memory for distances and indices
            std::unique_ptr<idx_t[]> coarse_ids(new idx_t[_nprobe]);
            std::unique_ptr<float[]> coarse_dists(new float[_nprobe]);

            // Use the quantizer to find the nprobe nearest centroids to the query
            _ivfpq->quantizer->search(1, q, _nprobe, coarse_dists.get(), coarse_ids.get());
            _ivfpq->invlists->prefetch_lists(coarse_ids.get(), _nprobe);

            // Compute inner_prod_table
            float* sim_table = new float[M * ksub];
            float* sim_table_2 = new float[M * ksub];
            _ivfpq->pq.compute_inner_prod_table(q, sim_table_2);

            // std::unique_ptr<faiss::InvertedListScanner> scanner(_ivfpq->get_InvertedListScanner(false, nullptr));
            // scanner->set_query(q);
            
            int total_access_count = 0;
            std::vector<int> top_candidates;
            float radius2 = radius * radius;

            for (idx_t list_no = 0; list_no < _nprobe; list_no++) {
                
                size_t list_size = _ivfpq->invlists->list_size(coarse_ids[list_no]);
    
                if (list_size == 0) {
                    continue;
                }

                total_access_count += list_size;

                // scanner->set_list(coarse_ids[list_no], coarse_dists[list_no]);

                // Calculate the distance table
                float dis0 = coarse_dists[list_no];
                idx_t coarse_id = coarse_ids[list_no];
                faiss::fvec_madd(
                    M * ksub,
                    _ivfpq->precomputed_table.data() + coarse_id * ksub * M,
                    -2.0,
                    sim_table_2,
                    sim_table);

                const uint8_t* codes = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_codes(coarse_id);
                const idx_t* ids = dynamic_cast<faiss::ArrayInvertedLists*>(_ivfpq->invlists)->get_ids(coarse_id);

                // Process codes in batches of 8
                for (size_t i = 0; i < list_size; i += 8) {
                    size_t batch_size = std::min<size_t>(8, list_size - i); // Handle the last batch if < 8

                    if (batch_size == 8) {
                        // std::cout << "batch 8"<< std::endl;
                        Dist8 dists;
                        const uint8_t* batch_codes = codes + i * code_size;
                        const idx_t* batch_ids = ids + i;

                        relaxed_lowerbound8(batch_codes, batch_ids, sim_table, dis0, dists);

                        for (size_t j = 0; j < 8; ++j) {
                            if (dists[j] <= radius2) {
                                top_candidates.push_back(batch_ids[j]);
                            }
                            // if(batch_ids[j] == 907264){
                            //     float dist_lq = std::sqrt(scanner->distance_to_code(batch_codes + j * code_size));
                            //     float dist_lx = _ivfpq->recons_errors[batch_ids[j]];
                            //     float lowerbound = relaxed_lowerbound(dist_lq, dist_lx);
                            //     std::cout << "Real Dist:" << _ivfpq->fvec_L2sqr(q, batch_ids[j]) << ", radius2:" << radius2 << std::endl;
                            //     std::cout << "My LB:" << dists[j] << ", Real LB:" << lowerbound << std::endl;
                            // }
                        }
                    } else {
                        for (size_t j = 0; j < batch_size; ++j) {
                            idx_t id = ids[i + j];
                            const uint8_t* code = codes + (i + j) * code_size;

                            float dist2_lq = dis0 + faiss::distance_single_code<faiss::PQDecoder8>(M, nbits, sim_table, code);
                            float dist_lq = std::sqrt(dist2_lq);
                            float dist_lx = _ivfpq->recons_errors[id];
                            float lowerbound = relaxed_lowerbound(dist_lq, dist_lx);

                            if (lowerbound <= radius2) {
                                top_candidates.push_back(id);
                            }
                        }
                    }
                }
            }  
                      
            int actual_access_count = top_candidates.size();
            for(size_t i=0; i<top_candidates.size(); i++){
                faiss::idx_t id = top_candidates[i];
                float dist = _ivfpq->fvec_L2sqr(q, id);

                if(dist <= radius2){
                    result.push_back(id);
                }  
            }
            
            _actual_distance_computation += actual_access_count;
            _total_distance_computation += total_access_count;
            _pruning_ratio += (1- 1.0*actual_access_count/total_access_count);

            delete[] sim_table;
            delete[] sim_table_2;
             
        }

        float relaxed_lowerbound(float a, float b) const {
            return (a - b) * (a - b) + 2 * _gamma * a * b;
        }

        void relaxed_lowerbound8(const uint8_t* codes, const idx_t* ids, float* dist_table, float dis0, Dist8& dists) const {
            return _relaxed_lowerbound8(codes, ids, dist_table, dis0, dists);
        }
  
        
#ifdef USE_AVX
  void _relaxed_lowerbound8(const uint8_t* codes, const idx_t* ids, float* dist_table, float dis0, Dist8& dists) const {
    // PQ distances
    float pq_dist[8] = { 
        0,
    };

    size_t M = _ivfpq->pq.M;
    size_t code_size = _ivfpq->code_size;
    size_t nbits = _ivfpq->pq.nbits;

    prefetch_l1(codes + 4 * code_size);
    prefetch_l1(codes + 5 * code_size);
    prefetch_l1(codes + 6 * code_size);
    prefetch_l1(codes + 7 * code_size);

    faiss::distance_four_codes<faiss::PQDecoder8>(
        M, nbits, dist_table,
        codes, codes + code_size, codes + 2 * code_size, codes + 3 * code_size,  //
        pq_dist[0], pq_dist[1], pq_dist[2], pq_dist[3]);

    // Prefetch reconstruciton errors
    prefetch_l1(_ivfpq->recons_errors.data() + ids[0]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[1]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[2]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[3]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[4]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[5]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[6]);
    prefetch_l1(_ivfpq->recons_errors.data() + ids[7]);

    faiss::distance_four_codes<faiss::PQDecoder8>(
        M, nbits, dist_table,                               //
        codes + 4 * code_size, codes + 5 * code_size,  //
        codes + 6 * code_size, codes + 7 * code_size,  //
        pq_dist[4], pq_dist[5], pq_dist[6], pq_dist[7]);

    // PQ distances
    __m256 vec_dis0 = _mm256_set1_ps(dis0);
    __m256 vec_pq_dist = _mm256_loadu_ps(pq_dist);
    vec_pq_dist = _mm256_add_ps(vec_pq_dist, vec_dis0);
    // _mm256_storeu_ps(pq_dist, vec_pq_dist);

    // lanmark distances
    __m256 vec_recons_error = _mm256_set_ps(_ivfpq->recons_errors[ids[7]], _ivfpq->recons_errors[ids[6]],  //
                                            _ivfpq->recons_errors[ids[5]], _ivfpq->recons_errors[ids[4]],  //
                                            _ivfpq->recons_errors[ids[3]], _ivfpq->recons_errors[ids[2]],  //
                                            _ivfpq->recons_errors[ids[1]], _ivfpq->recons_errors[ids[0]]);
    // Lowerbounds
    __m256 vec_lowerbounds = _relaxed_lowerbound8_avx2(_gamma, vec_pq_dist, vec_recons_error);
    _mm256_storeu_ps(dists.data(), vec_lowerbounds);
  }

  static __m256 _relaxed_lowerbound8_avx2(float gamma, __m256 pq_dist_vec,
                                          __m256 recons_error_vec) {
    // Squared root of PQ distances
    __m256 vec_a = _mm256_sqrt_ps(pq_dist_vec);
    // Reconstruction errors (actully squared roots of reconstruction errors)
    __m256 vec_b = recons_error_vec;
    // lowerbounds = (a[i] - b[i])^2 + 2 * gamma * a[i] * b[i]
    __m256 vec_diff = _mm256_sub_ps(vec_a, vec_b);
    __m256 vec_diff_squared = _mm256_mul_ps(vec_diff, vec_diff);
    __m256 vec_2gamma = _mm256_set1_ps(2 * gamma);
    __m256 vec_ab = _mm256_mul_ps(vec_a, vec_b);
    __m256 vec_2gamma_ab = _mm256_mul_ps(vec_2gamma, vec_ab);
    __m256 lowerbounds = _mm256_add_ps(vec_diff_squared, vec_2gamma_ab);
    return lowerbounds;
  }

#else
  void _relaxed_lowerbound8(const uint8_t* codes, const idx_t* ids, float* dist_table, float dis0, Dist8& dists) const {
    // PQ distances
    float pq_dist[8] = {0};

    size_t M = _ivfpq->pq.M;
    size_t code_size = _ivfpq->code_size;
    size_t nbits = _ivfpq->pq.nbits;
    
    // Compute PQ distances
    for(size_t i=0; i<8; i++){
        pq_dist[i] = dis0 + faiss::distance_single_code<faiss::PQDecoder8>(M, nbits, dist_table, codes + i*code_size);
    }
    
    // Load reconstruction errors
    const std::vector<float>& recons_errors = _ivfpq->recons_errors;

    // Compute lowerbounds
    for (int i = 0; i < 8; ++i) {
        float a = std::sqrt(pq_dist[i]);  // Squared root of PQ distances
        float b = recons_errors[ids[i]];  // Reconstruction errors (actually squared roots)
        float diff = a - b;
        float lowerbound = diff * diff + 2 * _gamma * a * b;
        dists[i] = lowerbound;
    }
  }
#endif
};
}
}