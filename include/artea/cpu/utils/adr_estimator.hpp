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

/*
 * @FilePath: /Artea/include/artea/cpu/utils/adr_estimator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: TBB-accelerated Average Distance Ratio (ADR) estimator
 *               for 1-NN evaluation. Mirrors the RecallEstimator layout
 *               (forward-declared in @c computer_traits.hpp, exposed
 *               via @c adr_estimator_t on @c ComputerTraits).
 *
 *               ADR is defined for 1-NN tasks as
 *                   ADR = (1 / |Q|) * Σ_q  d(q, retrieved_q) / d(q, true_NN_q)
 *               where @c retrieved_q is the top-1 vertex returned by the
 *               router and @c true_NN_q is the ground-truth nearest
 *               neighbour (GT column 0). ADR >= 1; exact retrieval gives
 *               ADR = 1. "Whatever @c dist_func returns" is treated as
 *               the distance — consistent with the project's existing
 *               ADR vs NDC profilers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <fmt/format.h>
#include <artea/common/logger.hpp>
#include <artea/cpu/router/data_structures/result_entry_concept.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT>
class ADREstimator final {

    using vertex_id_t    = typename ComputerTraitsT::vertex_id_t;
    using vertex_num_t   = typename ComputerTraitsT::vertex_num_t;
    using vec_ele_t      = typename ComputerTraitsT::vec_ele_t;
    using distance_t     = typename ComputerTraitsT::distance_t;
    using idlist_array_t = typename ComputerTraitsT::idlist_array_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using dist_func_t    = typename ComputerTraitsT::dist_func_t;

public:

    ADREstimator() = default;

    /**
     * @brief ADR@1 over a flat result vector (row-major @c num_queries *
     *        @p topk result entries). Only the position-0 entry per
     *        query is consumed. Queries whose GT distance is zero are
     *        skipped to avoid div-by-zero; the divisor is the count of
     *        queries that survived that guard.
     *
     * @tparam ResultEntryT  Anything satisfying @c ResultEntry (exposes
     *                       @c get_vid()); @c CandidateEntry qualifies.
     * @param predictions    Flat array sized @p num_queries * @p topk,
     *                       indexed as @c predictions[i * topk + j].
     * @param topk           Row stride; used only for indexing, ADR is
     *                       still defined on the top-1 entry.
     * @param gt_vecs        Ground-truth vid array; column 0 holds the
     *                       true NN for each query.
     * @param query_vecs     Query vectors, same order as @p predictions.
     * @param base_vecs      Base-dataset vectors the vids index into.
     * @param dist_func      Distance functor; used for both the
     *                       retrieved-NN distance and the GT distance
     *                       so the ratio is self-consistent regardless
     *                       of whether @c dist_func returns L2 or
     *                       squared-L2.
     * @return Averaged ratio >= 1 (0.0 when every query was skipped).
     */
    template <ResultEntry ResultEntryT>
    auto calculate_adr_at_1(
        const std::vector<ResultEntryT>& predictions,
        std::size_t                      topk,
        const idlist_array_t&            gt_vecs,
        const vector_array_t&            query_vecs,
        const vector_array_t&            base_vecs,
        const dist_func_t&               dist_func,
        std::size_t                      num_queries
    ) const -> double {
        if (num_queries == 0 || topk == 0) return 0.0;

        if (predictions.size() != num_queries * topk) {
            ARTEA_ERROR(fmt::format(
                "Predictions size mismatch: expected {}*{}={}, got {}",
                num_queries, topk, num_queries * topk, predictions.size()));
            return 0.0;
        }
        if (gt_vecs.get_num_vecs() < num_queries) {
            ARTEA_ERROR(fmt::format(
                "Ground truth has {} rows, need >= {}",
                gt_vecs.get_num_vecs(), num_queries));
            return 0.0;
        }
        if (gt_vecs.get_vec_dim() < 1) {
            ARTEA_ERROR("Ground truth dimension must be >= 1 for ADR@1");
            return 0.0;
        }

        struct Accum {
            double      sum_ratio = 0.0;
            std::size_t counted   = 0;
        };

        const Accum total = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, num_queries),
            Accum{},
            [&](const tbb::blocked_range<std::size_t>& r, Accum local) -> Accum {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t retrieved = predictions[i * topk].get_vid();
                    const vertex_id_t true_nn   = gt_vecs.get(i)[0];
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    const vec_ele_t* r_vec = base_vecs.get(retrieved);
                    const vec_ele_t* t_vec = base_vecs.get(true_nn);
                    const distance_t d_retrieved = dist_func(q_vec, r_vec);
                    const distance_t d_true      = dist_func(q_vec, t_vec);
                    if (d_true <= distance_t(0)) continue;
                    local.sum_ratio += static_cast<double>(d_retrieved) / static_cast<double>(d_true);
                    ++local.counted;
                }
                return local;
            },
            [](Accum a, Accum b) -> Accum {
                return Accum{a.sum_ratio + b.sum_ratio, a.counted + b.counted};
            }
        );

        if (total.counted == 0) return 0.0;
        return total.sum_ratio / static_cast<double>(total.counted);
    }

    /**
     * @brief ADR@1 over an @c idlist_array_t of predictions (one row per
     *        query, @c get_vec_dim() == topk). Convenience overload for
     *        call sites that hold vid lists directly.
     */
    auto calculate_adr_at_1(
        const idlist_array_t& predictions,
        const idlist_array_t& gt_vecs,
        const vector_array_t& query_vecs,
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func
    ) const -> double {
        const std::size_t num_queries = gt_vecs.get_num_vecs();
        if (num_queries == 0) return 0.0;

        if (predictions.get_num_vecs() != num_queries) {
            ARTEA_ERROR(fmt::format(
                "Predictions size mismatch: expected {} queries, got {}",
                num_queries, predictions.get_num_vecs()));
            return 0.0;
        }
        if (predictions.get_vec_dim() < 1 || gt_vecs.get_vec_dim() < 1) {
            ARTEA_ERROR("Predictions/GT dimensions must be >= 1 for ADR@1");
            return 0.0;
        }

        struct Accum {
            double      sum_ratio = 0.0;
            std::size_t counted   = 0;
        };

        const Accum total = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, num_queries),
            Accum{},
            [&](const tbb::blocked_range<std::size_t>& r, Accum local) -> Accum {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t retrieved = predictions.get(i)[0];
                    const vertex_id_t true_nn   = gt_vecs.get(i)[0];
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    const vec_ele_t* r_vec = base_vecs.get(retrieved);
                    const vec_ele_t* t_vec = base_vecs.get(true_nn);
                    const distance_t d_retrieved = dist_func(q_vec, r_vec);
                    const distance_t d_true      = dist_func(q_vec, t_vec);
                    if (d_true <= distance_t(0)) continue;
                    local.sum_ratio += static_cast<double>(d_retrieved) / static_cast<double>(d_true);
                    ++local.counted;
                }
                return local;
            },
            [](Accum a, Accum b) -> Accum {
                return Accum{a.sum_ratio + b.sum_ratio, a.counted + b.counted};
            }
        );

        if (total.counted == 0) return 0.0;
        return total.sum_ratio / static_cast<double>(total.counted);
    }

};  // class ADREstimator

}   // namespace cpu
}   // namespace artea
