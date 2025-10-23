/*
 * @FilePath: /Artea/include/artea/cpu/index_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-23 21:12:21
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

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/simd_distance.hpp>

namespace artea {
namespace cpu {

template <
    typename vecs_num_t, 
    typename vec_ele_t,
    typename vec_id_t = vecs_num_t
>
class MiniBatchKmeans {

public:
    MiniBatchKmeans(
        vecs_num_t num_vecs,
        cluster_num_t num_clusters, 
        vecs_num_t batch_size, 
        iter_t num_iters,
        vec_dim_t vec_dim,
        vec_ele_t tolerance = 1e-4
    ) : _num_vecs(num_vecs), 
        _num_clusters(num_clusters), 
        _batch_size(batch_size), 
        _num_iters(num_iters), 
        _vec_dim(vec_dim),
        _tolerance(tolerance)
    {
        _centroids = new VectorArray<cluster_num_t, vec_ele_t>(_num_clusters, vec_dim);
        _vecs_cluster_idx = new cluster_id_t[num_vecs];
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
    }

    /**
     * @brief Trains the centroids using the mini-batch algorithm.
     * This method only computes the final centroid locations. It does not assign cluster labels.
     * @param vecs_array Pointer to the input vector data.
     * @return The actual number of iterations performed before stopping.
     */
    auto fit(const VectorArray<vecs_num_t, vec_ele_t>* vecs_array) -> iter_t {
        // --- Step 1: Fast random initialization ---
        // We select k unique random points from the dataset as initial centroids.
        std::vector<vec_id_t> indices(_num_vecs);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (cluster_id_t i = 0; i < _num_clusters; ++i) {
            std::copy_n(vecs_array->get(indices[i]), _vec_dim, _centroids->get(i));
        }

        // --- Step 2: Iteratively update centroids with mini-batches ---
        std::vector<vecs_num_t> counts(_num_clusters, 0);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<vecs_num_t> dist(0, _num_vecs - 1);
        artea::cpu::SIMDDistance<vec_ele_t, artea::cpu::SIMDDistanceType::EUCLIDEAN> fast_distance(_vec_dim);

        // For early stopping
        auto old_centroids = new VectorArray<cluster_num_t, vec_ele_t>(_num_clusters, _vec_dim);

        for (iter_t iter = 0; iter < _num_iters; ++iter) {
            // Store current centroids to check for convergence later.
            std::copy_n(_centroids->get_all(), _num_clusters * _vec_dim, old_centroids->get_all());

            // Generate and process one mini-batch.
            for (vecs_num_t i = 0; i < _batch_size; ++i) {
                vec_id_t sample_idx = dist(rng);
                vec_ele_t* current_vec = vecs_array->get(sample_idx);
                cluster_id_t closest_cluster_id = 0;
                vec_ele_t min_dist = std::numeric_limits<vec_ele_t>::max();

                for (cluster_id_t c = 0; c < _num_clusters; ++c) {
                    vec_ele_t d = fast_distance(current_vec, _centroids->get(c));
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

            // --- Early Stopping Check ---
            vec_ele_t centroids_shift_sq = 0.0f;
            for (cluster_id_t c = 0; c < _num_clusters; ++c) {
                centroids_shift_sq += fast_distance(_centroids->get(c), old_centroids->get(c));
            }
            
            if (centroids_shift_sq < _tolerance) {
                delete old_centroids;
                return iter + 1; // Return the number of iterations completed.
            }
        }
        
        delete old_centroids;
        return _num_iters;
    }

    /**
     * @brief A convenient method to both train the model and assign cluster labels.
     * It first calls fit() to compute the centroids, then assigns labels to all data points.
     * @param vecs_array Pointer to the input vector data.
     */
    auto build_index(const VectorArray<vecs_num_t, vec_ele_t>* vecs_array) -> void {
        // Step 1: Train the centroids.
        fit(vecs_array);
        // Step 2: Assign all vectors to their final nearest cluster (in parallel).
        _assign_clusters(vecs_array);
    }

    auto get_num_clusters() -> cluster_num_t {
        return _num_clusters;
    }

    auto get_centroids() -> VectorArray<cluster_num_t, vec_ele_t>* {
        return _centroids;
    }

    auto get_vecs_cluster_idx() -> VectorArray<vecs_num_t, cluster_id_t>* {
        return _vecs_cluster_idx;
    }

private:
    /**
     * @brief Assigns each vector in the dataset to the nearest centroid. This method is parallelized.
     * @param vecs_array Pointer to the input vector data.
     */
    auto _assign_clusters(const VectorArray<vecs_num_t, vec_ele_t>* vecs_array) -> void {
        artea::cpu::SIMDDistance<vec_ele_t, artea::cpu::SIMDDistanceType::EUCLIDEAN> fast_distance(_vec_dim);

        #pragma omp parallel for schedule(static)
        for (vecs_num_t i = 0; i < _num_vecs; ++i) {
            vec_ele_t min_dist = std::numeric_limits<vec_ele_t>::max();
            cluster_id_t closest_cluster = 0;
            for (cluster_id_t c = 0; c < _num_clusters; ++c) {
                vec_ele_t dist = fast_distance(vecs_array->get(i), _centroids->get(c));
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_cluster = c;
                }
            }
            _vecs_cluster_idx[i] = closest_cluster;
        }
    }

private:
    /** @brief Number of vectors. */
    vecs_num_t _num_vecs;

    /** @brief Vector dimension. */
    vec_dim_t _vec_dim;

    /** @brief Size of each mini-batch. */
    vecs_num_t _batch_size;

    /** @brief Number of clusters. */
    cluster_num_t _num_clusters;

    /** @brief Max number of iterations. */
    iter_t _num_iters;

    /** @brief Threshold for early stopping. */
    vec_ele_t _tolerance;

    /** @brief Array of cluster centroids. */
    VectorArray<cluster_num_t, vec_ele_t>* _centroids;

    /** @brief Index of each vector in the cluster. */
    cluster_id_t* _vecs_cluster_idx;

};  // class MiniBatchKmeans

}   // namespace cpu
}   // namespace artea
