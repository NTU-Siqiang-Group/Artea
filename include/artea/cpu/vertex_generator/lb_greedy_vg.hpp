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
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class LBGreedyVG : public VertexGeneratorTraitsT::template vertex_generator_t<LBGreedyVG<VertexGeneratorTraitsT>> {

    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vertex_num_t = typename VertexGeneratorTraitsT::vertex_num_t;
    using distance_t = typename VertexGeneratorTraitsT::distance_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using dist_func_t = typename VertexGeneratorTraitsT::dist_func_t;
    using approx_rnet_t = typename VertexGeneratorTraitsT::approx_rnet_t;

    /**
     * @brief Candidate information for batch processing
     */
    struct CandidateInfo {
        vec_id_t vec_id;                // Vector ID in base_vecs
        distance_t min_dist_to_rnet;    // Minimum distance to existing approx_rnet
    };

public:
    LBGreedyVG(const dist_func_t& dist_func) : _dist_func(dist_func) {}

    /**
     * @brief Generate vertex IDs only (without vector data)
     * @return Vector of vertex IDs
     */
    auto gen_id_array(
        const vector_array_t& base_vecs,
        const distance_t min_radius,
        const vertex_num_t max_result_size,
        const vertex_num_t batch_size = 512,
        const vertex_num_t term_thresh = 17
    ) -> std::vector<vec_id_t> {
        const vec_num_t total_base_vecs = base_vecs.get_num_vecs();
        std::vector<vec_id_t> result_ids;

        if (total_base_vecs == 0) { return result_ids; }

        result_ids.reserve(max_result_size);
        result_ids.push_back(0);

        // Temporary vector array for distance computation
        vector_array_t temp_vectors(base_vecs.get_vec_dim());
        temp_vectors.reserve(max_result_size);
        temp_vectors.append_vec(base_vecs.get(0));

        vec_num_t batch_start = 0;

        while (result_ids.size() < max_result_size) {
            batch_start += batch_size;
            if (batch_start >= total_base_vecs) { break; }

            const vec_num_t batch_end = std::min(batch_start + batch_size, total_base_vecs);
            const vec_num_t current_batch_size = batch_end - batch_start;

            tbb::enumerable_thread_specific<std::vector<CandidateInfo>> thread_local_candidates;

            tbb::parallel_for(
                tbb::blocked_range<vec_num_t>(0, current_batch_size),
                [&](const tbb::blocked_range<vec_num_t>& r) {
                    auto& local_candidates = thread_local_candidates.local();

                    for (vec_num_t local_idx = r.begin(); local_idx != r.end(); ++local_idx) {
                        const vec_num_t candidate_idx = batch_start + local_idx;
                        const vec_ele_t* candidate_vec = base_vecs.get(candidate_idx);

                        distance_t min_distance = std::numeric_limits<distance_t>::max();
                        bool is_qualifying = true;

                        for (vec_num_t ret_idx = 0; ret_idx < temp_vectors.get_num_vecs(); ++ret_idx) {
                            distance_t dist = _dist_func(candidate_vec, temp_vectors.get(ret_idx));
                            if (dist < min_radius) {
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

            std::vector<CandidateInfo> qualifying_candidates;
            for (const auto& local_candidates : thread_local_candidates) {
                qualifying_candidates.insert(
                    qualifying_candidates.end(),
                    local_candidates.begin(),
                    local_candidates.end()
                );
            }

            std::sort(qualifying_candidates.begin(), qualifying_candidates.end(),
                [](const CandidateInfo& a, const CandidateInfo& b) {
                    return a.min_dist_to_rnet > b.min_dist_to_rnet;
                }
            );

            std::vector<const vec_ele_t*> batch_added_vecs;
            batch_added_vecs.reserve(qualifying_candidates.size());

            for (const auto& candidate : qualifying_candidates) {
                if (result_ids.size() >= max_result_size) { break; }

                const vec_ele_t* candidate_vec = base_vecs.get(candidate.vec_id);

                bool conflicts_with_batch = false;
                for (const vec_ele_t* batch_vec : batch_added_vecs) {
                    if (_dist_func(candidate_vec, batch_vec) < min_radius) {
                        conflicts_with_batch = true;
                        break;
                    }
                }

                if (!conflicts_with_batch) {
                    result_ids.push_back(candidate.vec_id);
                    temp_vectors.append_vec(candidate_vec);
                    batch_added_vecs.push_back(candidate_vec);
                }
            }

            /**
             * @brief Termination condition based on statistical coverage analysis.
             *
             * Let X be the number of qualifying candidates (uncovered points) in a batch.
             * X follows a binomial distribution X ~ B(n, p), where n is the batch size
             * and p is the uncovered rate (probability that a candidate is not yet covered
             * by the current r-net, i.e., distance to r-net ≥ min_radius).
             *
             * Using normal approximation: X ~ N(np, np(1-p)) for large n.
             *
             * Example with batch_size=512 and term_thresh=17:
             * - If p=5% (5% uncovered, 95% covered): μ=25.6, σ=4.93
             *   P(X ≥ 17) ≈ Φ(1.74) ≈ 96% confidence to continue sampling
             *
             * Interpretation: With term_thresh=17 and batch_size=512, the algorithm
             * continues with ~96% confidence when uncovered rate ≥5% (coverage ≤95%),
             * and terminates when coverage reaches ~96-97%, ensuring a dense r-net.
             *
             * NOTE: This check is performed AFTER merging the current batch to ensure
             * that qualifying candidates from this batch are not discarded.
             */
            if (qualifying_candidates.size() < term_thresh) { break; }
        }

        return result_ids;
    }

    /**
     * @brief Default generate method (returns ID array only)
     */
    auto generate(
        const vector_array_t& base_vecs,
        const distance_t min_radius,
        const vertex_num_t max_result_size,
        const vertex_num_t batch_size = 512,
        const vertex_num_t term_thresh = 17
    ) -> std::vector<vec_id_t> {
        return gen_id_array(base_vecs, min_radius, max_result_size, batch_size, term_thresh);
    }

private:
    const dist_func_t& _dist_func;

};

}   // namespace cpu
}   // namespace artea
