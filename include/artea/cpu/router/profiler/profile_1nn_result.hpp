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
 * @FilePath: /Artea/include/artea/cpu/router/profiler/profile_1nn_result.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Aggregate per-hop profiling result for pure greedy 1-NN
 *               routing. Returned by @c SLRouterProfiler::profile and
 *               @c HGRouterProfiler::profile.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace artea {
namespace cpu {

/**
 * @brief One hop's aggregate across the query set.
 *
 * @c avg_ndc   Mean cumulative number of distance computations across
 *              queries at this hop, using sticky-last-value for queries
 *              that have already terminated at a smaller hop index.
 * @c adr       Average Distance Ratio for 1-NN at this hop:
 *              @c (1/|Q|) * sum_q d(q, best_at_hop) / d(q, true_nn).
 *              Always >= 1; equals 1 iff every non-skipped query's
 *              retrieved vertex has the same distance as its true NN.
 */
struct HopProfilePoint {
    double avg_ndc;
    double adr;
};

/**
 * @brief Result of @c SLRouterProfiler::profile /
 *        @c HGRouterProfiler::profile.
 *
 * @c per_hop_statistics  Indexed by hop (0 = seed-only state). Length
 *                        equals the longest per-query trajectory; by
 *                        sticky-last-value, every index in range is
 *                        defined for every contributing query.
 * @c num_queries         Number of queries that contributed to the
 *                        aggregates (excludes @c skipped_queries).
 * @c skipped_queries     Queries dropped because @c d(q, true_nn) == 0
 *                        (ADR is undefined there; common when the query
 *                        set contains exact duplicates of base vectors).
 */
struct Profile1NNResult {
    std::vector<HopProfilePoint> per_hop_statistics;
    uint32_t num_queries = 0;
    uint32_t skipped_queries = 0;
};

}   // namespace cpu
}   // namespace artea
