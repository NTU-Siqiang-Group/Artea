/*
 * @FilePath: /Artea/include/artea/cpu/kmeans_partition.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-20 11:47:28
 * @Description: High-performance K-means implementation using Intel oneDAL (CPU only).
 */

#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <numeric>
#include <cmath>
#include <limits>
#include <atomic>

// Intel oneTBB headers for parallelism
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>

// Intel oneDAL headers
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/random_seq.hpp> // Added for high-performance MKL RNG
#include <artea/definitions.hpp>
#include <artea/config.hpp>
#include <artea/logger.hpp>

namespace artea {
namespace cpu {

namespace dal = oneapi::dal;

/**
 * @brief Structure to hold clustering evaluation metrics.
 *
 * Contains statistics about the quality of the partition, including inertia (SSE),
 * silhouette score, and load balance information.
 */
template <typename vertex_num_t, typename vec_ele_t>
struct PartitionMetrics {
    using distance_t = vec_ele_t;
    const part_num_t num_parts;

    explicit PartitionMetrics(const part_num_t parts) :
        num_parts(parts),
        intra_cluster_distance_avg(0),
        inter_cluster_distance_avg(0),
        silhouette_score(0),
        inertia(0),
        part_sizes(parts, 0),
        part_sizes_stddev(0.0f)
    {}

    distance_t intra_cluster_distance_avg; ///< Average distance from point to its centroid
    distance_t inter_cluster_distance_avg; ///< Average distance from point to the 2nd nearest centroid
    distance_t silhouette_score;           ///< Simplified Silhouette Score in range [-1, 1]
    distance_t inertia;                    ///< Sum of Squared Errors (SSE)
    std::vector<vertex_num_t> part_sizes;  ///< Number of points assigned to each cluster
    float part_sizes_stddev;               ///< Standard deviation of cluster sizes (indicates Load Imbalance)
};

/**
 * @brief High-performance K-means clustering wrapper using Intel oneDAL.
 *
 * This class handles model training, data partitioning, and reordering.
 * It leverages TBB for parallel data manipulation and oneDAL for vectorized K-means computation.
 */
template <typename vertex_num_t, typename vec_ele_t>
class KmeansPartition final {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;

public:
    /**
     * @brief Construct a new Kmeans Partition object.
     *
     * @param vertex_num Total number of vectors in the dataset.
     * @param cluster_num Target number of clusters (K).
     * @param threshold Convergence threshold for K-means (default: 1e-4).
     * @param max_iters Maximum number of iterations for Lloyd's algorithm (default: 100).
     */
    KmeansPartition(
        const vertex_num_t vertex_num,
        const vertex_num_t cluster_num,
        const distance_t threshold = 1e-4,
        const iter_t max_iters = 100
    ) : _vertex_num(vertex_num),
        _cluster_num(cluster_num),
        _threshold(threshold),
        _max_iters(max_iters)
    {
    }

    /**
     * @brief Trains the K-means model on a subset of the provided data.
     *
     * This function samples the input data using RandomSeq, wraps it into a oneDAL table,
     * and executes the training algorithm to find centroids.
     *
     * @param vecs_arr The source vector array containing all data points.
     * @param sampling_ratio Ratio of data to use for training (0.0 < ratio <= 1.0).
     */
    auto fit(
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        const float sampling_ratio = 0.1f
    ) -> void {

        // Generate a new VectorArray containing only the sampled subset of vectors
        auto sampled_vecs = _sampling(vecs_arr, sampling_ratio);
        auto sample_count = sampled_vecs.get_num_vecs();
        auto dim = sampled_vecs.get_vec_dim();

        logger.info(fmt::format("K-means Fit: Sampling {} vectors ({:.1f}%) for training.",
                                sample_count, sampling_ratio * 100));

        // Wrap data into oneDAL Homogen Table.
        // Since VectorArray guarantees contiguous memory, we use zero-copy wrapping.
        auto x_train = dal::homogen_table::wrap(
            sampled_vecs.get_all(),
            sample_count,
            dim
        );

        // Configure K-means descriptor with Lloyd's dense method
        const auto kmeans_desc = dal::kmeans::descriptor<distance_t, dal::kmeans::method::lloyd_dense>()
            .set_cluster_count(_cluster_num)
            .set_max_iteration_count(_max_iters)
            .set_accuracy_threshold(_threshold);

        try {
            // Execute training to compute centroids
            auto result = dal::train(kmeans_desc, x_train);
            _model = result.get_model();

            // Cache centroids in a standard vector for faster access during the evaluation phase
            auto centroids_table = _model.get_centroids();
            dal::row_accessor<const distance_t> acc(centroids_table);
            auto arr = acc.pull();
            const auto* data_ptr = arr.get_data();
            const auto count = arr.get_count();
            _centroids.assign(data_ptr, data_ptr + count);

        } catch (const std::exception& e) {
            logger.error(fmt::format("Kmeans fit failed: {}", e.what()));
            throw;
        }
    }

    /**
     * @brief Partitions the entire dataset and physically reorders vectors in memory.
     *
     * This process involves:
     * 1. Assigning cluster labels to all vectors (Inference).
     * 2. Calculating partition sizes in parallel (Chunked Histogram).
     * 3. Computing write offsets (Prefix Sum).
     * 4. Moving data to the new sorted location in parallel (Scatter).
     *
     * @param vecs_arr The original vector array.
     * @return std::pair containing:
     *         - A new VectorArray with vectors grouped by cluster ID.
     *         - A vector of offsets indicating the start index of each cluster.
     * @throws std::runtime_error If the model has not been trained (fit) yet.
     */
    auto partition_and_reorder(const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr)
        -> std::pair<VectorArray<vertex_num_t, vec_ele_t>, std::vector<vertex_num_t>>
    {
        if (_centroids.empty()) {
            throw std::runtime_error("Model not trained. Call fit() first.");
        }

        const vertex_num_t num_vecs = vecs_arr.get_num_vecs();
        const auto dim = vecs_arr.get_vec_dim();

        // Wrap the full dataset for inference
        auto x_full = dal::homogen_table::wrap(vecs_arr.get_all(), num_vecs, dim);

        // Configure descriptor for inference (0 iterations implies assignment only)
        const auto kmeans_desc = dal::kmeans::descriptor<distance_t, dal::kmeans::method::lloyd_dense>()
            .set_cluster_count(_cluster_num)
            .set_max_iteration_count(0)
            .set_accuracy_threshold(0.0);

        // Perform inference to get cluster assignments for every point
        auto result = dal::infer(kmeans_desc, _model, x_full);

        auto responses = result.get_responses();
        dal::row_accessor<const int> response_acc(responses);
        auto labels_raw = response_acc.pull(); // Access raw integer labels

        // --- Parallel Partitioning Preparation (Chunked Histogram) ---

        // Determine chunking strategy to maximize TBB thread utilization
        const size_t num_threads = tbb::this_task_arena::max_concurrency();
        const size_t num_chunks = num_threads * 4; // Oversubscribe to balance load
        const size_t chunk_size = (static_cast<size_t>(num_vecs) + num_chunks - 1) / num_chunks;

        // Allocate storage for local histograms: [chunk_id][cluster_id]
        std::vector<std::vector<size_t>> chunk_counts(num_chunks, std::vector<size_t>(_cluster_num, 0));

        // Compute local counts for each chunk in parallel
        // This avoids atomic operations on a global counter array
        tbb::parallel_for(size_t(0), num_chunks, [&](size_t chunk_idx) {
            vertex_num_t start = static_cast<vertex_num_t>(chunk_idx * chunk_size);
            vertex_num_t end = std::min(static_cast<vertex_num_t>(start + chunk_size), num_vecs);

            auto& local_counts = chunk_counts[chunk_idx];
            for (vertex_num_t i = start; i < end; ++i) {
                int label = labels_raw[i];
                local_counts[label]++;
            }
        });

        // --- Serial Prefix Sum (Calculate Offsets) ---

        // final_offsets: Global start index for each cluster in the new array
        std::vector<vertex_num_t> final_offsets(_cluster_num, 0);

        // chunk_write_pos: Specific write start index for each cluster within each chunk
        // This allows threads to write to independent memory locations later
        std::vector<std::vector<vertex_num_t>> chunk_write_pos(num_chunks, std::vector<vertex_num_t>(_cluster_num, 0));

        vertex_num_t accumulated_pos = 0;

        // Iterate by Cluster first, then by Chunk to ensure clusters are contiguous in the final array
        for (vertex_num_t k = 0; k < _cluster_num; ++k) {
            final_offsets[k] = accumulated_pos;

            for (size_t c = 0; c < num_chunks; ++c) {
                chunk_write_pos[c][k] = accumulated_pos;
                accumulated_pos += static_cast<vertex_num_t>(chunk_counts[c][k]);
            }
        }

        // --- Parallel Reordering (Scatter Copy) ---

        VectorArray<vertex_num_t, vec_ele_t> reordered_vecs(num_vecs, dim);
        vec_ele_t* dest_base = reordered_vecs.get_all();
        const vec_ele_t* src_base = vecs_arr.get_all();

        // Each chunk copies its assigned vectors to their pre-calculated destinations
        tbb::parallel_for(size_t(0), num_chunks, [&](size_t chunk_idx) {
            vertex_num_t start = static_cast<vertex_num_t>(chunk_idx * chunk_size);
            vertex_num_t end = std::min(static_cast<vertex_num_t>(start + chunk_size), num_vecs);

            // Use a local copy of write positions to track current insertion index
            auto local_write_pos = chunk_write_pos[chunk_idx];

            for (vertex_num_t i = start; i < end; ++i) {
                int label = labels_raw[i];
                vertex_num_t dest_idx = local_write_pos[label]++; // Get position and increment

                const vec_ele_t* src_ptr = src_base + static_cast<size_t>(i) * dim;
                vec_ele_t* dest_ptr = dest_base + static_cast<size_t>(dest_idx) * dim;

                std::copy(src_ptr, src_ptr + dim, dest_ptr);
            }
        });

        return {std::move(reordered_vecs), final_offsets};
    }

    /**
     * @brief Evaluates the quality of the clustering result.
     *
     * Computes metrics such as Inertia, Silhouette Score, and Load Imbalance using
     * TBB parallel reduction for efficiency.
     *
     * @param vecs_arr The dataset to evaluate against the trained centroids.
     * @return PartitionMetrics structure containing the calculated statistics.
     */
    auto evaluate(const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr)
        -> PartitionMetrics<vertex_num_t, vec_ele_t>
    {
        PartitionMetrics<vertex_num_t, vec_ele_t> metrics(_cluster_num);

        if (_centroids.empty()) {
            logger.warn("Evaluate called before fit. Returning empty metrics.");
            return metrics;
        }

        const vertex_num_t num_vecs = vecs_arr.get_num_vecs();
        const auto dim = vecs_arr.get_vec_dim();

        struct ThreadResult {
            distance_t inertia = 0.0;
            distance_t silhouette_sum = 0.0;
            distance_t intra_sum = 0.0;
            distance_t inter_sum = 0.0;
            std::vector<vertex_num_t> counts;

            explicit ThreadResult(std::size_t k) : counts(k, 0) {}
        };

        auto final_res = tbb::parallel_reduce(
            tbb::blocked_range<vertex_num_t>(0, num_vecs),
            ThreadResult(_cluster_num),
            [&](const tbb::blocked_range<vertex_num_t>& r, ThreadResult res) -> ThreadResult {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* vec = vecs_arr.get(i);

                    // Find 1st and 2nd nearest centroids
                    distance_t min_dist = std::numeric_limits<distance_t>::max();
                    distance_t second_min_dist = std::numeric_limits<distance_t>::max();
                    part_num_t best_k = 0;

                    for (part_num_t k = 0; k < _cluster_num; ++k) {
                        distance_t dist_sq = 0.0;
                        const vec_ele_t* center = &_centroids[static_cast<std::size_t>(k) * dim];

                        for (uint32_t d = 0; d < dim; ++d) {
                            distance_t diff = static_cast<distance_t>(vec[d]) - static_cast<distance_t>(center[d]);
                            dist_sq += diff * diff;
                        }

                        if (dist_sq < min_dist) {
                            second_min_dist = min_dist;
                            min_dist = dist_sq;
                            best_k = k;
                        } else if (dist_sq < second_min_dist) {
                            second_min_dist = dist_sq;
                        }
                    }

                    res.inertia += min_dist;
                    res.counts[best_k]++;

                    distance_t a = std::sqrt(min_dist);
                    distance_t b = std::sqrt(second_min_dist);

                    distance_t s = 0.0;
                    if (b > 1e-9) {
                        s = (b - a) / std::max(a, b);
                    }

                    res.silhouette_sum += s;
                    res.intra_sum += a;
                    res.inter_sum += b;
                }
                return res;
            },
            [&](ThreadResult a, const ThreadResult& b) -> ThreadResult {
                a.inertia += b.inertia;
                a.silhouette_sum += b.silhouette_sum;
                a.intra_sum += b.intra_sum;
                a.inter_sum += b.inter_sum;
                for (std::size_t k = 0; k < _cluster_num; ++k) {
                    a.counts[k] += b.counts[k];
                }
                return a;
            }
        );

        metrics.inertia = static_cast<vec_ele_t>(final_res.inertia);
        metrics.silhouette_score = static_cast<vec_ele_t>(final_res.silhouette_sum / num_vecs);
        metrics.intra_cluster_distance_avg = static_cast<vec_ele_t>(final_res.intra_sum / num_vecs);
        metrics.inter_cluster_distance_avg = static_cast<vec_ele_t>(final_res.inter_sum / num_vecs);
        metrics.part_sizes = final_res.counts;

        distance_t mean = static_cast<distance_t>(num_vecs) / _cluster_num;
        distance_t accum = 0.0;
        for (auto sz : metrics.part_sizes) {
            accum += (sz - mean) * (sz - mean);
        }
        metrics.part_sizes_stddev = static_cast<float>(std::sqrt(accum / _cluster_num));

        return metrics;
    }

    __attribute__((always_inline))
    auto get_centroids() const -> const std::vector<vec_ele_t>& {
        return _centroids;
    }

private:

    /**
     * @brief Creates a new VectorArray containing a random subset of the source data.
     *
     * Utilizes RandomSeq (Intel MKL) to efficiently generate random indices.
     * Uses TBB for parallel gathering of vectors.
     *
     * @param source_arr The full dataset.
     * @param sampling_ratio Percentage of data to sample.
     * @return A new VectorArray instance with the sampled vectors.
     */
    auto _sampling(
        const VectorArray<vertex_num_t, vec_ele_t>& source_arr,
        const float sampling_ratio
    ) -> VectorArray<vertex_num_t, vec_ele_t> {

        vertex_num_t sample_count = static_cast<vertex_num_t>(_vertex_num * sampling_ratio);

        // Constraints: Ensure minimum sample size covers all clusters
        if (sample_count < _cluster_num) {
            sample_count = std::min(_cluster_num, _vertex_num);
        }

        // Create the destination container
        auto dim = source_arr.get_vec_dim();
        VectorArray<vertex_num_t, vec_ele_t> sampled_arr(sample_count, dim);

        vec_ele_t* dest_base = sampled_arr.get_all();
        const vec_ele_t* src_base = source_arr.get_all();

        // --- Index Generation ---

        // Generate random indices efficiently using MKL
        std::vector<vertex_id_t> indices(sample_count);
        // MKL-based high-performance random number generator
        RandomSeq<vertex_num_t> _rng(_vertex_num);
        _rng.generate(indices, sample_count);

        // Gather vectors in parallel based on generated indices
        tbb::parallel_for(tbb::blocked_range<vertex_num_t>(0, sample_count),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (auto i = r.begin(); i != r.end(); ++i) {
                    vertex_id_t src_idx = indices[i];

                    const vec_ele_t* src_ptr = src_base + static_cast<size_t>(src_idx) * dim;
                    vec_ele_t* dest_ptr = dest_base + static_cast<size_t>(i) * dim;

                    std::copy(src_ptr, src_ptr + dim, dest_ptr);
                }
            }
        );

        return sampled_arr;
    }

    const vertex_num_t _vertex_num;
    const vertex_num_t _cluster_num;
    const distance_t _threshold;
    const iter_t _max_iters;

    // oneDAL model object
    dal::kmeans::model<dal::kmeans::task::clustering> _model;

    // Local cache of centroids (Layout: K x Dim)
    std::vector<vec_ele_t> _centroids;

};  // class KmeansPartition

}   // namespace cpu
}   // namespace artea