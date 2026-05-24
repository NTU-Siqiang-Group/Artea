// Copyright 2026 Weitang Ye
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

#pragma once

#include <algorithm>
#include <limits>
#include <vector>
#include <cmath>
#include <atomic>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/enumerable_thread_specific.h>
#include <boost/math/distributions/binomial.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Mini-Batch Greedy Vertex Generator with High-Precision Sampling
 *
 * MBGreedyVG uses a mini-batch greedy approach with statistical termination:
 * - Processes candidates in small batches
 * - Selects the candidate with maximum distance to existing r-net in each batch
 * - Terminates when k consecutive batches find no qualifying candidates
 *
 * The parameter k is computed based on coverage_ratio and confidence to ensure
 * high-precision sampling with statistical guarantees.
 */
template <typename VertexGeneratorTraitsT, typename DistFuncT>
class MBGreedyVG :
    public VertexGeneratorTraitsT::template vertex_generator_t<MBGreedyVG<VertexGeneratorTraitsT, DistFuncT>> {

    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vertex_num_t = typename VertexGeneratorTraitsT::vertex_num_t;
    using distance_t = typename VertexGeneratorTraitsT::distance_t;
    using ratio_t = typename VertexGeneratorTraitsT::ratio_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using dist_func_t = DistFuncT;
    using approx_rnet_t = typename VertexGeneratorTraitsT::approx_rnet_t;

public:
    explicit MBGreedyVG(const dist_func_t& dist_func) : _dist_func(dist_func) {}

    /**
     * @brief Compute the number of consecutive empty batches (k) needed for termination
     *
     * Statistical Formulation:
     * Let p = 1 - coverage_ratio be the uncovered rate. When actual coverage < target,
     * the probability of a batch being empty (all candidates covered) is (1-p)^n,
     * where n = batch_size.
     *
     * For k consecutive empty batches, the probability of premature termination is:
     * P(k consecutive empty batches | coverage < target) = (1-p)^(n·k)
     *
     * We want this false-positive rate to be at most α = 1 - confidence:
     * (1-p)^(n·k) ≤ 1 - confidence
     * => n·k·log(1-p) ≤ log(1 - confidence)
     * => k ≥ log(1 - confidence) / [n · log(1 - p)]
     *
     * Example: coverage_ratio=0.95, confidence=0.99, batch_size=64
     * - p = 0.05, (1-p)^n = 0.95^64 ≈ 0.0374
     * - k = ceil(log(0.01) / (64 · log(0.95))) = ceil(-4.605 / -3.28) ≈ 2
     * - Verification: 0.0374^2 ≈ 0.0014 < 0.01 ✓
     *
     * @param coverage_ratio Target coverage ratio (e.g., 0.95 for 95% coverage)
     * @param confidence Confidence level (e.g., 0.99 for 99% confidence)
     * @param batch_size Batch size for processing
     * @return Number of consecutive empty batches needed
     */
    static auto compute_empty_batch_count(
        const ratio_t coverage_ratio,
        const ratio_t confidence,
        const vertex_num_t batch_size
    ) -> vertex_num_t {
        if (coverage_ratio <= 0.0 || coverage_ratio >= 1.0) {
            ARTEA_ERROR("coverage_ratio must be in (0, 1)");
        }
        if (confidence <= 0.0 || confidence >= 1.0) {
            ARTEA_ERROR("confidence must be in (0, 1)");
        }
        if (batch_size < 1) {
            ARTEA_ERROR("batch_size must be at least 1");
        }

        double p = 1.0 - static_cast<double>(coverage_ratio);
        double log_alpha = std::log(1.0 - static_cast<double>(confidence));
        double log_1_minus_p = std::log(1.0 - p);

        // k ≥ log(1 - confidence) / [n · log(1 - p)]
        double k_double = log_alpha / (static_cast<double>(batch_size) * log_1_minus_p);

        // Use ceiling to be conservative (require more empty batches)
        vertex_num_t k = static_cast<vertex_num_t>(std::ceil(k_double));

        // Ensure k is at least 1
        return std::max(static_cast<vertex_num_t>(1), k);
    }

    /**
     * @brief Generate approximate r-net with auto-computed empty_batch_count
     *
     * This overload automatically computes the number of consecutive empty batches
     * needed for termination based on coverage_ratio and confidence.
     *
     * @param vecs_data The vector array to generate vertices from
     * @param rnet_radius Minimum distance between vertices
     * @param max_result_size Maximum number of vertices to generate
     * @param coverage_ratio Target coverage ratio (e.g., 0.95 for 95% coverage)
     * @param confidence Confidence level (e.g., 0.96 for 96% confidence)
     * @param batch_size Batch size for processing
     * @return Approximate r-net (VertexSubset)
     */
    auto generate_impl(
        const vector_array_t& vecs_data,
        const distance_t rnet_radius,
        const vertex_num_t max_result_size,
        const ratio_t coverage_ratio,
        const ratio_t confidence,
        const vertex_num_t batch_size
    ) -> approx_rnet_t {
        vertex_num_t empty_batch_count = compute_empty_batch_count(coverage_ratio, confidence, batch_size);
        return generate_impl(vecs_data, rnet_radius, max_result_size, batch_size, empty_batch_count);
    }

    /**
     * @brief Generate approximate r-net (manual mode)
     *
     * Manually specify batch_size and empty_batch_count.
     *
     * MBGreedyVG uses a Mini-Batch greedy approach with statistical termination:
     * - Processes candidates in small batches
     * - Selects the candidate with maximum distance to existing r-net in each batch
     * - Terminates when k consecutive batches find no qualifying candidates
     *
     * @param vecs_data The vector array to generate vertices from
     * @param rnet_radius Minimum distance between vertices
     * @param max_result_size Maximum number of vertices to generate
     * @param batch_size Batch size for processing
     * @param empty_batch_count Number of consecutive empty batches before termination
     * @return Approximate r-net (VertexSubset)
     */
    auto generate_impl(
        const vector_array_t& vecs_data,
        const distance_t rnet_radius,
        const vertex_num_t max_result_size,
        const vertex_num_t batch_size,
        const vertex_num_t empty_batch_count
    ) -> approx_rnet_t {
        const vec_num_t total_vecs = vecs_data.get_num_vecs();

        approx_rnet_t approx_rnet(vecs_data.get_vec_dim());

        if (total_vecs == 0 || max_result_size == 0) { return approx_rnet; }

        approx_rnet.reserve(max_result_size);

        // Select first vertex
        approx_rnet.vec_ids.push_back(0);
        approx_rnet.vecs_data.append_vec(vecs_data.get(0));

        vec_num_t batch_start = 0;
        vertex_num_t consecutive_empty_batches = 0;

        /**
         * @brief Sequential batch sampling strategy
         *
         * This algorithm samples consecutive batches of vectors from the dataset
         * (batch_start to batch_start + batch_size). This sequential sampling is
         * statistically valid ONLY because the input dataset has been shuffled.
         *
         * IMPORTANT: The caller MUST shuffle the dataset before calling this function.
         * Without shuffling, sequential sampling would introduce spatial bias and
         * violate the statistical assumptions of the termination condition.
         */
        while (approx_rnet.get_num_vecs() < max_result_size) {
            if (batch_start >= total_vecs) { break; }

            const vec_num_t batch_end = std::min(batch_start + batch_size, total_vecs);
            const vec_num_t current_batch_size = batch_end - batch_start;

            std::vector<distance_t> candidate_min_distances(current_batch_size);
            std::fill(candidate_min_distances.begin(), candidate_min_distances.end(),
                     std::numeric_limits<distance_t>::max());

            const vec_num_t rnet_size = approx_rnet.vecs_data.get_num_vecs();

            /**
             * @brief 2D parallelization over (candidate × r-net) using blocked_range2d
             * Avoids integer division/modulo. Uses global covered flags for cross-range early exit.
             */

            std::vector<std::atomic<bool>> candidate_covered(current_batch_size);
            for (vec_num_t i = 0; i < current_batch_size; ++i) {
                candidate_covered[i].store(false, std::memory_order_relaxed);
            }

            tbb::enumerable_thread_specific<std::vector<distance_t>> thread_local_mins(
                std::vector<distance_t>(current_batch_size, std::numeric_limits<distance_t>::max())
            );

            tbb::parallel_for(
                tbb::blocked_range2d<vec_num_t>(0, current_batch_size, 0, rnet_size),
                [&](const tbb::blocked_range2d<vec_num_t>& r) {
                    auto& local_mins = thread_local_mins.local();

                    for (vec_num_t local_idx = r.rows().begin(); local_idx != r.rows().end(); ++local_idx) {
                        // Early exit: if this candidate is already covered globally
                        if (candidate_covered[local_idx].load(std::memory_order_relaxed)) {
                            continue;
                        }

                        const vec_num_t candidate_idx = batch_start + local_idx;
                        const vec_ele_t* candidate_vec = vecs_data.get(candidate_idx);

                        for (vec_num_t ret_idx = r.cols().begin(); ret_idx != r.cols().end(); ++ret_idx) {
                            const vec_ele_t* rnet_vec = approx_rnet.vecs_data.get(ret_idx);

                            distance_t dist = _dist_func(candidate_vec, rnet_vec);
                            local_mins[local_idx] = std::min(local_mins[local_idx], dist);

                            // Mark as covered and break if distance < radius
                            if (local_mins[local_idx] < rnet_radius) {
                                candidate_covered[local_idx].store(true, std::memory_order_relaxed);
                                break;
                            }
                        }
                    }
                }
            );

            // Reduce thread-local minimums to global minimums
            for (const auto& local_mins : thread_local_mins) {
                for (vec_num_t local_idx = 0; local_idx < current_batch_size; ++local_idx) {
                    candidate_min_distances[local_idx] = std::min(
                        candidate_min_distances[local_idx],
                        local_mins[local_idx]
                    );
                }
            }

            const auto max_it = std::max_element(candidate_min_distances.begin(), candidate_min_distances.end());
            const distance_t max_distance = *max_it;

            /**
             * @brief Termination condition based on consecutive empty batches
             *
             * If max_distance < rnet_radius, no qualifying candidates exist in this batch.
             * We count consecutive empty batches and terminate when the count reaches
             * empty_batch_count (k).
             *
             * Statistical Interpretation:
             * Let p = 1 - coverage_ratio be the uncovered rate.
             * For a batch of size n, P(batch is empty | coverage achieved) = (1 - p)^n
             * For k consecutive batches: P(all k empty) = (1 - p)^(n·k)
             *
             * Example with batch_size=64, coverage_ratio=0.95, confidence=0.96:
             * - p = 0.05 (5% uncovered)
             * - P(one batch empty) = 0.95^64 ≈ 0.0374
             * - k = ceil(log(0.96) / (64 · log(0.95))) ≈ 1
             *
             * Example with batch_size=32, coverage_ratio=0.95, confidence=0.96:
             * - p = 0.05 (5% uncovered)
             * - P(one batch empty) = 0.95^32 ≈ 0.1934
             * - k = ceil(log(0.96) / (32 · log(0.95))) ≈ 3
             *
             * Smaller batches require more consecutive empty batches for the same confidence.
             */
            if (max_distance < rnet_radius) {
                consecutive_empty_batches++;
                if (consecutive_empty_batches >= empty_batch_count) {
                    break;
                }
            } else {
                // Reset counter when we find a qualifying candidate
                consecutive_empty_batches = 0;

                const vec_num_t best_local_idx = std::distance(candidate_min_distances.begin(), max_it);
                const vec_id_t best_vec_id = batch_start + best_local_idx;
                approx_rnet.vec_ids.push_back(best_vec_id);
                approx_rnet.vecs_data.append_vec(vecs_data.get(best_vec_id));
            }

            batch_start += batch_size;
        }

        return approx_rnet;
    }

private:
    const dist_func_t& _dist_func;

};

}   // namespace cpu
}   // namespace artea
