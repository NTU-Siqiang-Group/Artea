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
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <boost/math/distributions/normal.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class LBGreedyVG : public VertexGeneratorTraitsT::template vertex_generator_t<LBGreedyVG<VertexGeneratorTraitsT>> {

    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vertex_num_t = typename VertexGeneratorTraitsT::vertex_num_t;
    using distance_t = typename VertexGeneratorTraitsT::distance_t;
    using ratio_t = typename VertexGeneratorTraitsT::ratio_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using dist_func_t = typename VertexGeneratorTraitsT::dist_func_t;
    using approx_rnet_t = typename VertexGeneratorTraitsT::approx_rnet_t;

public:
    explicit LBGreedyVG(const dist_func_t& dist_func) : _dist_func(dist_func) {}

    /**
     * @brief Compute sampling_batch_size and term_thresh from coverage_ratio and confidence
     *
     * Uses the formula: term_thresh = np - z·√(np(1-p))
     * where n is sampling_batch_size, p is uncovered_rate = 1 - coverage_ratio,
     * and z is the confidence z-value from standard normal distribution.
     *
     * @param coverage_ratio Target coverage ratio (e.g., 0.95 for 95% coverage)
     * @param confidence Confidence level (e.g., 0.96 for 96% confidence)
     * @param sampling_batch_size Desired batch size (e.g., 512, 1024, 2048)
     * @return Computed term_thresh value
     */
    static auto compute_term_thresh(
        const ratio_t coverage_ratio,
        const ratio_t confidence,
        const vertex_num_t sampling_batch_size
    ) -> vertex_num_t {
        if (coverage_ratio <= 0.0 || coverage_ratio >= 1.0) {
            logger.error("coverage_ratio must be in (0, 1)");
        }
        if (confidence <= 0.0 || confidence >= 1.0) {
            logger.error("confidence must be in (0, 1)");
        }
        if (sampling_batch_size < 1) {
            logger.error("sampling_batch_size must be at least 1");
        }

        // Calculate uncovered rate
        double p = 1.0 - static_cast<double>(coverage_ratio);

        // Calculate z-value for confidence
        boost::math::normal_distribution<double> normal(0.0, 1.0);
        double z_value = boost::math::quantile(normal, static_cast<double>(confidence));

        // Calculate term_thresh: μ - z·σ = np - z·√(np(1-p))
        double n = static_cast<double>(sampling_batch_size);
        double mean = n * p;
        double std_dev = std::sqrt(n * p * (1.0 - p));
        double term_thresh_double = mean - z_value * std_dev;

        // Ensure term_thresh is at least 1
        vertex_num_t term_thresh = static_cast<vertex_num_t>(std::max(1.0, std::floor(term_thresh_double)));

        return term_thresh;
    }

    /**
     * @brief Generate approximate r-net with auto-computed term_thresh
     *
     * This overload automatically computes term_thresh based on the desired
     * coverage ratio and confidence level.
     *
     * @param vecs_data The vector array to generate vertices from
     * @param rnet_radius Minimum distance between vertices
     * @param max_result_size Maximum number of vertices to generate
     * @param coverage_ratio Target coverage ratio (e.g., 0.95 for 95% coverage)
     * @param confidence Confidence level (e.g., 0.96 for 96% confidence)
     * @param sampling_batch_size Batch size for processing
     * @return Approximate r-net (VertexSubset)
     */
    auto generate_impl(
        const vector_array_t& vecs_data,
        const distance_t rnet_radius,
        const vertex_num_t max_result_size,
        const ratio_t coverage_ratio,
        const ratio_t confidence,
        const vertex_num_t sampling_batch_size
    ) -> approx_rnet_t {
        vertex_num_t term_thresh = compute_term_thresh(coverage_ratio, confidence, sampling_batch_size);
        return generate_impl(vecs_data, rnet_radius, max_result_size, sampling_batch_size, term_thresh);
    }

    /**
     * @brief Generate approximate r-net (manual mode)
     *
     * Manually specify sampling_batch_size and term_thresh.
     *
     * @param vecs_data The vector array to generate vertices from
     * @param rnet_radius Minimum distance between vertices
     * @param max_result_size Maximum number of vertices to generate
     * @param sampling_batch_size Batch size for processing
     * @param term_thresh Termination threshold
     * @return Approximate r-net (VertexSubset)
     */
    auto generate_impl(
        const vector_array_t& vecs_data,
        const distance_t rnet_radius,
        const vertex_num_t max_result_size,
        const vertex_num_t sampling_batch_size,
        const vertex_num_t term_thresh
    ) -> approx_rnet_t {
        const vec_num_t total_vecs = vecs_data.get_num_vecs();

        approx_rnet_t approx_rnet(vecs_data.get_vec_dim());

        if (total_vecs == 0 || max_result_size == 0) { return approx_rnet; }

        approx_rnet.reserve(max_result_size);

        // Select first vertex
        approx_rnet.vec_ids.push_back(0);
        approx_rnet.vecs_data.append_vec(vecs_data.get(0));

        vec_num_t batch_start = 0;

        // Pre-allocate thread-local storage and reusable buffers outside the loop
        tbb::enumerable_thread_specific<std::vector<std::pair<vec_id_t, distance_t>>> thread_local_candidates;
        std::vector<std::pair<vec_id_t, distance_t>> qualifying_candidates;
        std::vector<const vec_ele_t*> batch_added_vecs;

        qualifying_candidates.reserve(sampling_batch_size);
        batch_added_vecs.reserve(sampling_batch_size);

        while (approx_rnet.get_num_vecs() < max_result_size) {
            if (batch_start >= total_vecs) { break; }

            const vec_num_t batch_end = std::min(batch_start + sampling_batch_size, total_vecs);
            const vec_num_t current_batch_size = batch_end - batch_start;

            // Clear thread-local vectors (preserve capacity for reuse)
            for (auto& local_vec : thread_local_candidates) {
                local_vec.clear();
            }

            tbb::parallel_for(
                tbb::blocked_range<vec_num_t>(0, current_batch_size),
                [&](const tbb::blocked_range<vec_num_t>& r) {
                    auto& local_candidates = thread_local_candidates.local();

                    for (vec_num_t local_idx = r.begin(); local_idx != r.end(); ++local_idx) {
                        // Get actual candidate index
                        vec_num_t candidate_idx = batch_start + local_idx;

                        const vec_ele_t* candidate_vec = vecs_data.get(candidate_idx);

                        distance_t min_distance = std::numeric_limits<distance_t>::max();
                        bool is_qualifying = true;

                        for (vec_num_t ret_idx = 0; ret_idx < approx_rnet.vecs_data.get_num_vecs(); ++ret_idx) {
                            distance_t dist = _dist_func(candidate_vec, approx_rnet.vecs_data.get(ret_idx));
                            if (dist < rnet_radius) {
                                is_qualifying = false;
                                break;
                            }
                            min_distance = std::min(min_distance, dist);
                        }

                        if (is_qualifying) {
                            local_candidates.push_back({candidate_idx, min_distance});
                        }
                    }
                }
            );

            // Reuse qualifying_candidates vector
            qualifying_candidates.clear();
            for (const auto& local_candidates : thread_local_candidates) {
                qualifying_candidates.insert(
                    qualifying_candidates.end(),
                    local_candidates.begin(),
                    local_candidates.end()
                );
            }

            std::sort(qualifying_candidates.begin(), qualifying_candidates.end(),
                [](const auto& a, const auto& b) {
                    return a.second > b.second;
                }
            );

            // Reuse batch_added_vecs vector
            batch_added_vecs.clear();

            _inner_batch_filter(
                vecs_data,
                qualifying_candidates,
                rnet_radius,
                max_result_size,
                approx_rnet,
                batch_added_vecs
            );

            /**
             * @brief Termination condition based on statistical coverage analysis.
             *
             * Let X be the number of qualifying candidates (uncovered points) in a batch.
             * X follows a binomial distribution X ~ B(n, p), where n is the batch size
             * and p is the uncovered rate (probability that a candidate is not yet covered
             * by the current r-net, i.e., distance to r-net ≥ rnet_radius).
             *
             * Using normal approximation: X ~ N(np, np(1-p)) for large n.
             *
             * Example with sampling_batch_size=512 and term_thresh=17:
             * - If p=5% (5% uncovered, 95% covered): μ=25.6, σ=4.93
             *   P(X ≥ 17) ≈ Φ(1.75) ≈ 96% confidence to continue sampling
             *
             * Interpretation: With term_thresh=17 and sampling_batch_size=512, the algorithm
             * continues with ~96% confidence when uncovered rate ≥5% (coverage ≤95%),
             * and terminates when coverage reaches ~95-96%, ensuring a dense r-net.
             *
             * Parameter Selection Guide (for 95% coverage target, 96% confidence):
             * ┌────────────┬──────────┬─────────────┬────────────────────────┐
             * │ Batch Size │ μ (mean) │ σ (std dev) │ term_thresh (96% conf) │
             * ├────────────┼──────────┼─────────────┼────────────────────────┤
             * │    512     │  25.6    │    4.93     │          16            │
             * │   1024     │  51.2    │    6.97     │          38            │
             * │   2048     │  102.4   │    9.86     │          85            │
             * └────────────┴──────────┴─────────────┴────────────────────────┘
             *
             * Formula: term_thresh = floor(μ - z·σ) = floor(np - z·√(np(1-p)))
             *   where z ≈ 1.75 for 96% confidence
             *   and p = 0.05 for 95% coverage target (5% uncovered rate)
             *   Note: floor() is used for conservative termination (samples longer)
             *
             * NOTE: This check is performed AFTER merging the current batch to ensure
             * that qualifying candidates from this batch are not discarded.
             */
            if (qualifying_candidates.size() < term_thresh) { break; }

            batch_start += sampling_batch_size;
        }

        return approx_rnet;
    }

private:
    const dist_func_t& _dist_func;

    /**
     * @brief Greedily insert qualifying candidates from current batch
     * @param vecs_data Source vector array
     * @param qualifying_candidates Sorted candidates (by distance, descending)
     * @param rnet_radius Minimum distance threshold
     * @param max_result_size Maximum number of vertices to generate
     * @param approx_rnet Output r-net (modified in-place)
     * @param batch_added_vecs Vectors added in this batch (modified in-place)
     */
    __attribute__((always_inline)) inline auto _inner_batch_filter(
        const vector_array_t& vecs_data,
        const std::vector<std::pair<vec_id_t, distance_t>>& qualifying_candidates,
        const distance_t rnet_radius,
        const vertex_num_t max_result_size,
        approx_rnet_t& approx_rnet,
        std::vector<const vec_ele_t*>& batch_added_vecs
    ) const -> void {
        for (const auto& [candidate_id, candidate_dist] : qualifying_candidates) {
            if (approx_rnet.get_num_vecs() >= max_result_size) { break; }

            const vec_ele_t* candidate_vec = vecs_data.get(candidate_id);

            bool conflicts_with_batch = false;
            for (const vec_ele_t* batch_vec : batch_added_vecs) {
                if (_dist_func(candidate_vec, batch_vec) < rnet_radius) {
                    conflicts_with_batch = true;
                    break;
                }
            }

            if (!conflicts_with_batch) {
                approx_rnet.vec_ids.push_back(candidate_id);
                approx_rnet.vecs_data.append_vec(candidate_vec);
                batch_added_vecs.push_back(candidate_vec);
            }
        }
    }

};

}   // namespace cpu
}   // namespace artea
