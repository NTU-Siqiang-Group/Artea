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
 * @FilePath: /Artea/include/artea/cpu/partitioning/cluster_evaluator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Generic Cluster Evaluator independent of the clustering algorithm.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Structure to hold clustering evaluation metrics.
 *
 * Contains statistical information about the clustering quality, including
 * intra/inter-cluster distances, silhouette score, and partition balance.
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

    /** @brief Average intra-cluster distance */
    distance_t intra_cluster_distance_avg;

    /** @brief Average inter-cluster distance */
    distance_t inter_cluster_distance_avg;

    /** @brief Simplified Silhouette Score in range [-1, 1] */
    distance_t silhouette_score;

    /** @brief Sum of Squared Errors (SSE) */
    distance_t inertia;

    /** @brief Sizes of each partition */
    std::vector<vertex_num_t> part_sizes;

    /** @brief Standard deviation of partition sizes (Load Imbalance) */
    float part_sizes_stddev;
};

/**
 * @brief Generic evaluator for clustering results.
 *
 * Calculates metrics such as Inertia, Silhouette Score, and Load Balance
 * given a dataset and a set of centroids.
 *
 * @tparam vertex_num_t Integer type for vertex indices.
 * @tparam vec_ele_t Float type for vector elements.
 * @tparam DistanceFunc Functor type for distance calculation.
 */
template <typename vertex_num_t, typename vec_ele_t, typename DistanceFunc>
class ClusterEvaluator {
    using distance_t = vec_ele_t;

public:
    /**
     * @brief Construct a new Cluster Evaluator.
     * @param dist_func Reference to the distance calculation functor.
     */
    explicit ClusterEvaluator(const DistanceFunc& dist_func)
        : _dist_func(dist_func) {}

    /**
     * @brief Computes evaluation metrics for a given dataset and centroids.
     *
     * @param vecs_arr The source data vectors.
     * @param centroids The cluster centroids.
     * @return PartitionMetrics calculated metrics.
     */
    auto evaluate(
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        const VectorArray<vertex_num_t, vec_ele_t>& centroids
    ) const -> PartitionMetrics<vertex_num_t, vec_ele_t> {

        vertex_num_t num_vecs = vecs_arr.get_num_vecs();
        vertex_num_t num_clusters = centroids.get_num_vecs();
        PartitionMetrics<vertex_num_t, vec_ele_t> metrics(static_cast<part_num_t>(num_clusters));

        if (num_clusters == 0) {
            logger.warn("Evaluator received empty centroids. Returning empty metrics.");
            return metrics;
        }

        struct ThreadResult {
            distance_t inertia = 0.0;
            distance_t silhouette_sum = 0.0;
            distance_t intra_sum = 0.0;
            distance_t inter_sum = 0.0;
            std::vector<vertex_num_t> counts;
            explicit ThreadResult(std::size_t k) : counts(k, 0) {}
        };

        // Parallel reduction to compute metrics over the entire dataset
        auto final_res = tbb::parallel_reduce(
            tbb::blocked_range<vertex_num_t>(0, num_vecs),
            ThreadResult(num_clusters),
            [&](const tbb::blocked_range<vertex_num_t>& r, ThreadResult res) -> ThreadResult {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* vec = vecs_arr.get(i);

                    distance_t min_dist_sq = std::numeric_limits<distance_t>::max();
                    distance_t second_min_dist_sq = std::numeric_limits<distance_t>::max();
                    int best_k = 0;

                    for (int k = 0; k < num_clusters; ++k) {
                        const vec_ele_t* center = centroids.get(k);

                        // Use Injected Distance Functor
                        distance_t dist_sq = _dist_func(vec, center);

                        if (dist_sq < min_dist_sq) {
                            second_min_dist_sq = min_dist_sq;
                            min_dist_sq = dist_sq;
                            best_k = k;
                        } else if (dist_sq < second_min_dist_sq) {
                            second_min_dist_sq = dist_sq;
                        }
                    }

                    res.inertia += min_dist_sq;
                    res.counts[best_k]++;

                    distance_t a = std::sqrt(min_dist_sq);
                    distance_t b = std::sqrt(second_min_dist_sq);

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
                for (std::size_t k = 0; k < num_clusters; ++k) {
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

        distance_t mean = static_cast<distance_t>(num_vecs) / num_clusters;
        distance_t accum = 0.0;
        for (auto sz : metrics.part_sizes) {
            accum += (sz - mean) * (sz - mean);
        }
        metrics.part_sizes_stddev = static_cast<float>(std::sqrt(accum / num_clusters));

        return metrics;
    }

    __attribute__((always_inline))
    auto operator()(
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        const VectorArray<vertex_num_t, vec_ele_t>& centroids
    ) const -> PartitionMetrics<vertex_num_t, vec_ele_t> {
        return evaluate(vecs_arr, centroids);
    }

private:
    const DistanceFunc& _dist_func;

};

} // namespace cpu
} // namespace artea