/*
 * @FilePath: /Artea/include/artea/cpu/mini_batch_kmeans.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-24 11:03:11
 * @Date: 2025-10-23 13:00:00
 * @Description: 
 */

#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <limits>
#include <omp.h>

// --- Faiss Headers ---
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>

#include <artea/types.hpp>
#include <artea/logger.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/simd_distance.hpp>

namespace artea {
namespace cpu {

enum class AssignMethod {
    BRUTE_FORCE,
    FAISS_HNSW
};

template <
    typename vec_num_t, 
    typename vec_ele_t,
    SIMDDistanceType simd_distance_type = SIMDDistanceType::EUCLIDEAN,
    typename vec_id_t = vec_num_t
>
class MiniBatchKmeans {

public:
    MiniBatchKmeans(
        vec_num_t num_vecs,
        cluster_num_t num_clusters, 
        vec_num_t batch_size, 
        iter_t num_iters,
        vec_dim_t vec_dim,
        vec_ele_t tolerance = 1e-4
    ) : _num_vecs(num_vecs), 
        _num_clusters(num_clusters), 
        _batch_size(batch_size), 
        _num_iters(num_iters), 
        _vec_dim(vec_dim),
        _tolerance(tolerance),
        _logger("MiniBatchKmeans", system_log_level) // Initialize logger here
    {
        _centroids = new VectorArray<cluster_num_t, vec_ele_t>(_num_clusters, vec_dim);
        _vecs_cluster_idx = new cluster_id_t[num_vecs];
        _distance_calculator = new SIMDDistance<vec_ele_t, simd_distance_type>(vec_dim);
    }

    ~MiniBatchKmeans() {
        if (_centroids != nullptr) {
            delete _centroids;
            _centroids = nullptr;
        }
        if (_vecs_cluster_idx != nullptr) {
            delete _vecs_cluster_idx;
            _vecs_cluster_idx = nullptr;
        }
        if (_distance_calculator != nullptr) {
            delete _distance_calculator;
            _distance_calculator = nullptr;
        }
    }

    template <uint32_t log_interval = 10>
    auto fit(const VectorArray<vec_num_t, vec_ele_t>* vecs_array) -> iter_t {
        std::vector<vec_id_t> indices(_num_vecs);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (cluster_id_t i = 0; i < _num_clusters; ++i) {
            std::copy_n(vecs_array->get(indices[i]), _vec_dim, _centroids->get(i));
        }

        std::vector<vec_num_t> counts(_num_clusters, 0);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<vec_num_t> dist(0, _num_vecs - 1);
        
        auto old_centroids = new VectorArray<cluster_num_t, vec_ele_t>(_num_clusters, _vec_dim);

        _logger.info("Start training MiniBatchKmeans...");

        for (iter_t iter = 0; iter < _num_iters; ++iter) {
            std::copy_n(_centroids->get_all(), _num_clusters * _vec_dim, old_centroids->get_all());

            for (vec_num_t i = 0; i < _batch_size; ++i) {
                vec_id_t sample_idx = dist(rng);
                const vec_ele_t* current_vec = vecs_array->get(sample_idx);
                cluster_id_t closest_cluster_id = 0;
                vec_ele_t min_dist = std::numeric_limits<vec_ele_t>::max();

                for (cluster_id_t c = 0; c < _num_clusters; ++c) {
                    vec_ele_t d = (*_distance_calculator)(current_vec, _centroids->get(c));
                    if (d < min_dist) {
                        min_dist = d;
                        closest_cluster_id = c;
                    }
                }

                counts[closest_cluster_id]++;
                vec_ele_t learning_rate = 1.0f / counts[closest_cluster_id];
                vec_ele_t* centroid_to_update = _centroids->get(closest_cluster_id);
                
                for (vec_dim_t d = 0; d < _vec_dim; ++d) {
                    centroid_to_update[d] = (1.0f - learning_rate) * centroid_to_update[d] + learning_rate * current_vec[d];
                }
            }

            vec_ele_t centroids_shift_sq = 0.0f;
            for (cluster_id_t c = 0; c < _num_clusters; ++c) {
                centroids_shift_sq += (*_distance_calculator)(_centroids->get(c), old_centroids->get(c));
            }
            
            if (iter % log_interval == 0) {
                _logger.info(std::format("Iteration {}: Centroid shift^2 = {}", iter, centroids_shift_sq));
            }
            if (centroids_shift_sq < _tolerance) {
                _logger.success(std::format("Converged after {} iterations.", iter + 1));
                delete old_centroids;
                return iter + 1;
            }
        }
        
        _logger.warn(std::format("Reached max iterations ({}) without convergence.", _num_iters));
        delete old_centroids;
        return _num_iters;
    }

    template <AssignMethod M = AssignMethod::BRUTE_FORCE>
    auto assign_index(const VectorArray<vec_num_t, vec_ele_t>* vecs_array) -> void {
        if constexpr (M == AssignMethod::BRUTE_FORCE) {
            _brute_force_assign(vecs_array);
        } else if constexpr (M == AssignMethod::FAISS_HNSW) {
            _faiss_ann_assign(vecs_array);
        }
    }

    template <AssignMethod M = AssignMethod::BRUTE_FORCE>
    auto build_index(const VectorArray<vec_num_t, vec_ele_t>* vecs_array) -> void {
        fit(vecs_array);
        assign_index<M>(vecs_array);
    }

    auto get_num_clusters() const -> cluster_num_t { return _num_clusters; }
    auto get_centroids() const -> const VectorArray<cluster_num_t, vec_ele_t>* { return _centroids; }
    auto get_vecs_cluster_idx() const -> const cluster_id_t* { return _vecs_cluster_idx; }

private:

    auto _brute_force_assign(const VectorArray<vec_num_t, vec_ele_t>* vecs_array) -> void {
        _logger.info("Assigning clusters using Brute Force method (SIMD)...");
        #pragma omp parallel for schedule(static)
        for (vec_num_t i = 0; i < _num_vecs; ++i) {
            vec_ele_t min_dist = std::numeric_limits<vec_ele_t>::max();
            cluster_id_t closest_cluster = 0;
            for (cluster_id_t c = 0; c < _num_clusters; ++c) {
                vec_ele_t dist = (*_distance_calculator)(vecs_array->get(i), _centroids->get(c));
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_cluster = c;
                }
            }
            _vecs_cluster_idx[i] = closest_cluster;
        }
    }

    auto _faiss_ann_assign(const VectorArray<vec_num_t, vec_ele_t>* vecs_array) -> void {
        _logger.info("Assigning clusters using Faiss HNSW method...");
        faiss::IndexHNSWFlat index(_vec_dim, 32, faiss::METRIC_L2);
        index.add(_num_clusters, _centroids->get_all());

        std::vector<faiss::idx_t> labels(_num_vecs);
        std::vector<float> distances(_num_vecs); 
        index.search(_num_vecs, vecs_array->get_all(), 1, distances.data(), labels.data());

        #pragma omp parallel for schedule(static)
        for (vec_num_t i = 0; i < _num_vecs; ++i) {
            _vecs_cluster_idx[i] = static_cast<cluster_id_t>(labels[i]);
        }
    }

    /** @brief Number of vectors in the dataset. */
    vec_num_t _num_vecs;
    /** @brief Dimensionality of each vector. */
    vec_dim_t _vec_dim;
    /** @brief Batch size for MiniBatchKmeans. */
    vec_num_t _batch_size;
    /** @brief Number of clusters. */
    cluster_num_t _num_clusters;
    /** @brief Maximum number of iterations. */
    iter_t _num_iters;
    /** @brief Tolerance for convergence. */
    vec_ele_t _tolerance;
    /** @brief Centroids of the clusters. */
    VectorArray<cluster_num_t, vec_ele_t>* _centroids;
    /** @brief Index of each vector in the cluster. */
    cluster_id_t* _vecs_cluster_idx;
    /** @brief Distance calculator for MiniBatchKmeans. */
    SIMDDistance<vec_ele_t, simd_distance_type>* _distance_calculator;
    /** @brief Logger for MiniBatchKmeans. */
    ArteaLogger _logger;
};

} // namespace cpu
} // namespace artea