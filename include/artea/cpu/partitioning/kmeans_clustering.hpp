// Copyright 2025 Weitang Ye
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * @FilePath: /Artea/include/artea/cpu/partitioning/kmeans_clustering.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: High-performance K-means implementation delegating search to VectorRouter.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstring>

// Intel oneTBB headers
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <tbb/enumerable_thread_specific.h>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/utils/random_seq.hpp>
#include <artea/cpu/partitioning/vector_router.hpp>
#include <artea/cpu/partitioning/bruteforce_router.hpp>
#include <artea/cpu/utils/vector_sampler.hpp>
#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/cpu/partitioning/static_clustering.hpp>
#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief High-performance K-means clustering implementation.
 *
 * Utilizes SIMD-aligned memory structures and TBB parallelization to perform
 * fast clustering.
 *
 * This class acts as a coordinator (Controller). It manages the lifecycle of centroids
 * and orchestrates the EM (Expectation-Maximization) steps. The heavy lifting of
 * finding the nearest centroid (E-Step) is delegated to the injected `cluster_router_t`.
 *
 * @tparam vertex_num_t Integer type for vertex indices.
 * @tparam vec_ele_t Float type for vector elements.
 * @tparam cluster_router_t The router strategy (e.g., BruteforceRouter) used for nearest neighbor search.
 * @tparam dist_func_t Functor type for distance calculation.
 */
template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename dist_func_t = SIMDDistance<vec_ele_t, distance_metrics_t::EUCLIDEAN>,
    typename cluster_router_t = BruteforceRouter<vertex_num_t, vec_ele_t, dist_func_t, false>,
    typename vector_sampler_t = VectorSampler<vertex_num_t, vec_ele_t>
>
class KmeansClustering final :
    public StaticClustering<vertex_num_t, vec_ele_t, dist_func_t, cluster_router_t, vector_sampler_t,
        KmeansClustering<vertex_num_t, vec_ele_t, dist_func_t, cluster_router_t, vector_sampler_t>>
{

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using base_class_t = StaticClustering<vertex_num_t,  vec_ele_t, dist_func_t, cluster_router_t, vector_sampler_t,
        KmeansClustering<vertex_num_t, vec_ele_t, dist_func_t, cluster_router_t, vector_sampler_t>>;

public:
    /**
     * @brief Construct a new Kmeans Partition object.
     *
     * @param num_vertices Total number of vectors expected in the dataset.
     * @param num_clusters Target number of clusters (K).
     * @param dist_func Reference to an initialized SIMD-capable distance calculator.
     * @param threshold Convergence threshold for centroid shift.
     * @param max_iters Maximum number of Lloyd's iterations.
     */
    KmeansClustering(
        const vertex_num_t num_vertices,
        const cluster_num_t num_clusters,
        const vec_dim_t vec_dim,
        const dist_func_t& dist_func,
        const vector_sampler_t& sampler,
        const distance_t threshold = 1e-3,
        const iter_t max_iters = 100
    ) :
        base_class_t(num_vertices, num_clusters, vec_dim, dist_func, sampler),
        _threshold(threshold),
        _max_iters(max_iters)
    {
    }

    /**
     * @brief Trains the K-means model using the provided vector data.
     *
     * Performs random sampling to initialize centroids, then executes Lloyd's
     * algorithm until convergence or maximum iterations are reached.
     *
     * @param vecs_arr The source vector data container.
     * @param sampling_ratio Ratio of data to use for training (0.0 to 1.0).
     */
    auto fit_impl(
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        const float sampling_ratio = 0.1f
    ) -> void {
        // 1. Sampling
        // Randomly sample a subset of vectors to reduce computational load.
        auto sampled_vecs = this->_sampler(vecs_arr, sampling_ratio);
        auto sample_count = sampled_vecs.get_num_vecs();
        const vec_ele_t* data = sampled_vecs.get_all(); // Raw pointer for accumulation logic

        logger.info(fmt::format("K-means Fit: Training on {} vectors ({:.1f}%), Dim={}, K={}",
                                sample_count, sampling_ratio * 100, this->_vec_dim, this->_num_clusters));

        // 2. Initialization
        // Initialize centroids using the first K samples (simplest approach).
        this->_centroids = VectorArray<vertex_num_t, vec_ele_t>(this->_num_clusters, this->_vec_dim);
        vertex_num_t init_count = std::min(static_cast<vertex_num_t>(this->_num_clusters), sample_count);

        tbb::parallel_for(tbb::blocked_range<vertex_num_t>(0, init_count),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (auto i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* src = sampled_vecs.get(i);
                    vec_ele_t* dst = this->_centroids.get(i);
                    std::copy(src, src + this->_vec_dim, dst);
                }
            }
        );


        if (init_count < this->_num_clusters) {
            for (vertex_num_t i = init_count; i < this->_num_clusters; ++i) {
                const vec_ele_t* src = this->_centroids.get(i % init_count);
                vec_ele_t* dst = this->_centroids.get(i);
                std::copy(src, src + this->_vec_dim, dst);
            }
        }

        // Prepare temporary storage
        std::vector<vec_ele_t> new_centroids(this->_num_clusters * this->_vec_dim);
        std::vector<vertex_num_t> counts(this->_num_clusters);

        // This vector holds the cluster ID for each sampled vector
        std::vector<cluster_id_t> labels(sample_count);

        // 3. Lloyd's Algorithm Loop
        for (iter_t iter = 0; iter < _max_iters; ++iter) {
            distance_t total_shift = 0.0;

            // --- E-Step: Assignment ---
            // Construct the Router with the CURRENT centroids.
            // The router_t type encapsulates the parallel strategy (Bruteforce, HNSW, etc.)
            // and the intra-query parallelism config.

            this->_router.initialize(); // Essential if the router builds an index (e.g., HNSW)

            // Delegate the heavy search task to the router.
            // batch_query handles TBB parallelism internally.
            labels = this->_router.batch_query(sampled_vecs);

            // --- M-Step: Update Centroids ---
            // Reset accumulators
            std::fill(new_centroids.begin(), new_centroids.end(), 0);
            std::fill(counts.begin(), counts.end(), 0);

            // Thread-local accumulation (Map phase)
            struct CentroidAccumulator {
                std::vector<vec_ele_t> sum;
                std::vector<vertex_num_t> count;
                CentroidAccumulator(size_t k, size_t d) : sum(k * d, 0), count(k, 0) {}
            };

            tbb::enumerable_thread_specific<CentroidAccumulator> tls_acc(
                this->_num_clusters, this->_vec_dim
            );

            tbb::parallel_for(tbb::blocked_range<vertex_num_t>(0, sample_count),
                [&](const tbb::blocked_range<vertex_num_t>& r) {
                    auto& acc = tls_acc.local();
                    auto& local_sum = acc.sum;
                    auto& local_count = acc.count;

                    for (auto i = r.begin(); i != r.end(); ++i) {
                        cluster_id_t label = labels[i];
                        local_count[label]++;

                        const vec_ele_t* vec = data + static_cast<size_t>(i) * this->_vec_dim;
                        vec_ele_t* cent_sum = &local_sum[static_cast<size_t>(label) * this->_vec_dim];

                        #pragma omp simd
                        for (size_t d = 0; d < this->_vec_dim; ++d) {
                            cent_sum[d] += vec[d];
                        }
                    }
                }
            );

            // Reduce phase
            for (const auto& acc : tls_acc) {
                for (size_t k = 0; k < this->_num_clusters; ++k) {
                    counts[k] += acc.count[k];
                    size_t offset = k * this->_vec_dim;
                    const vec_ele_t* src_sum = &acc.sum[offset];
                    vec_ele_t* dst_sum = &new_centroids[offset];

                    #pragma omp simd
                    for (size_t d = 0; d < this->_vec_dim; ++d) {
                        dst_sum[d] += src_sum[d];
                    }
                }
            }

            // Final Update & Shift Calculation
            for (size_t k = 0; k < this->_num_clusters; ++k) {
                if (counts[k] > 0) {
                    distance_t inv_count = 1.0f / counts[k];
                    distance_t centroid_sq_diff = 0.0;

                    vec_ele_t* current_centroid_ptr = this->_centroids.get(k);
                    const vec_ele_t* accumulated_sum_ptr = &new_centroids[k * this->_vec_dim];

                    for (size_t d = 0; d < this->_vec_dim; ++d) {
                        vec_ele_t new_val = accumulated_sum_ptr[d] * inv_count;
                        vec_ele_t old_val = current_centroid_ptr[d];
                        vec_ele_t diff = new_val - old_val;

                        centroid_sq_diff += diff * diff;
                        current_centroid_ptr[d] = new_val;
                    }
                    total_shift += centroid_sq_diff;
                }
            }

            if (total_shift < _threshold) {
                logger.debug(fmt::format("K-means converged at iteration {} (Shift: {:e})", iter, total_shift));
                break;
            }
        }
    }

    /** @note Function partition_and_reorder is no longer supported in KMeansClustering class
     *        due to design changes. Please use the Partitioner class for data partitioning and
     *        reordering based on trained centroids.
     */

    #if 0
    /**
     * @brief Partitions the input dataset based on trained centroids and physically reorders memory.
     *
     * @param vecs_arr The input vector array.
     * @return std::pair A pair containing the reordered VectorArray and a list of offsets.
     */
    auto partition_and_reorder(const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr)
        -> std::pair<VectorArray<vertex_num_t, vec_ele_t>, std::vector<vertex_num_t>>
    {
        if (this->_centroids.get_num_vecs() == 0) {
            logger.error("Model not trained. Call fit() first.");
            throw std::runtime_error("Model not trained. Call fit() first.");
        }

        const vertex_num_t num_vecs = vecs_arr.get_num_vecs();

        // 1. Assign Labels using the VectorRouter
        std::vector<cluster_id_t> labels;

        this->_router.initialize();
        // This leverages the highly optimized batch_query implementation in VectorRouter
        labels = this->_router.batch_query(vecs_arr);

        // 2. Count Histogram (Parallel)
        const size_t num_threads = tbb::this_task_arena::max_concurrency();
        const size_t num_chunks = num_threads * 4;
        const size_t chunk_size = (static_cast<size_t>(num_vecs) + num_chunks - 1) / num_chunks;

        std::vector<std::vector<size_t>> chunk_counts(num_chunks, std::vector<size_t>(this->_num_clusters, 0));

        tbb::parallel_for(size_t(0), num_chunks, [&](size_t chunk_idx) {
            vertex_num_t start = static_cast<vertex_num_t>(chunk_idx * chunk_size);
            vertex_num_t end = std::min(static_cast<vertex_num_t>(start + chunk_size), num_vecs);

            auto& local_counts = chunk_counts[chunk_idx];
            for (vertex_num_t i = start; i < end; ++i) {
                local_counts[labels[i]]++;
            }
        });

        // 3. Compute Offsets (Serial prefix sum)
        std::vector<vertex_num_t> final_offsets(this->_num_clusters, 0);
        std::vector<std::vector<vertex_num_t>> chunk_write_pos(num_chunks, std::vector<vertex_num_t>(this->_num_clusters, 0));
        vertex_num_t accumulated_pos = 0;

        for (vertex_num_t k = 0; k < this->_num_clusters; ++k) {
            final_offsets[k] = accumulated_pos;
            for (size_t c = 0; c < num_chunks; ++c) {
                chunk_write_pos[c][k] = accumulated_pos;
                accumulated_pos += static_cast<vertex_num_t>(chunk_counts[c][k]);
            }
        }

        // 4. Scatter / Reorder Data (Parallel)
        VectorArray<vertex_num_t, vec_ele_t> reordered_vecs(num_vecs, this->_vec_dim);
        vec_ele_t* dest_base = reordered_vecs.get_all();
        const vec_ele_t* src_base = vecs_arr.get_all();

        tbb::parallel_for(size_t(0), num_chunks, [&](size_t chunk_idx) {
            vertex_num_t start = static_cast<vertex_num_t>(chunk_idx * chunk_size);
            vertex_num_t end = std::min(static_cast<vertex_num_t>(start + chunk_size), num_vecs);
            auto local_write_pos = chunk_write_pos[chunk_idx];

            for (vertex_num_t i = start; i < end; ++i) {
                cluster_id_t label = labels[i];
                vertex_num_t dest_idx = local_write_pos[label]++;

                const vec_ele_t* src_ptr = src_base + static_cast<size_t>(i) * this->_vec_dim;
                vec_ele_t* dest_ptr = dest_base + static_cast<size_t>(dest_idx) * this->_vec_dim;
                std::copy(src_ptr, src_ptr + this->_vec_dim, dest_ptr);
            }
        });

        return {std::move(reordered_vecs), final_offsets};
    }
    #endif

    /** --- Accessors --- **/

    __attribute__((always_inline))
    auto get_threshold() const -> const distance_t {
        return _threshold;
    }

    __attribute__((always_inline))
    auto get_max_iters() const -> const iter_t {
        return _max_iters;
    }

    __attribute__((always_inline))
    auto set_threshold(const distance_t threshold) -> void {
        _threshold = threshold;
    }

    __attribute__((always_inline))
    auto set_max_iters(const iter_t max_iters) -> void {
        _max_iters = max_iters;
    }

private:

    /** @brief Convergence threshold for centroid shift. */
    distance_t _threshold;

    /** @brief Maximum number of Lloyd's iterations. */
    iter_t _max_iters;

};  // class KmeansClustering

}   // namespace cpu
}   // namespace artea