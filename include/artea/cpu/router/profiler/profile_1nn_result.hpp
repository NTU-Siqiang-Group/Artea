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
 * @Description: Per-query 1-NN profiling result types:
 *                 - Profile1NNResult: raw (NDC, ADR) trajectories, one
 *                   per sampled query, used by
 *                   @c profile_adr_vs_ndc.
 *                 - LatencyProfileResult: per-query wall-clock
 *                   latencies + pXX summary, used by
 *                   @c profile_latency.
 */

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace artea {
namespace cpu {

/**
 * @brief Raw per-query 1-NN profiling result.
 *
 * @c trajectories    One entry per sampled query, in the order the
 *                    caller passed them in. Each inner vector holds
 *                    @c (cumulative_NDC, ADR) pairs — one per
 *                    successful cursor move (plus the seed). Skipped
 *                    queries (those with @c d(q, true_nn) == 0) keep
 *                    an empty trajectory.
 * @c num_skipped     Count of skipped queries within the sample.
 */
struct Profile1NNResult {
    using trajectory_t = std::vector<std::pair<uint64_t, double>>;
    std::vector<trajectory_t> trajectories;
    uint32_t num_skipped = 0;
};

/**
 * @brief Per-query wall-clock latency result + summary percentiles.
 *
 * @c latencies_us   Per-query latency in microseconds, one per query
 *                   in the query set (in query_vid order).
 * @c num_queries    Total queries measured (== @c latencies_us.size()).
 * @c p50_us .. p99_us Summary percentiles over @c latencies_us
 *                   (linear index, nearest-rank).
 */
struct LatencyProfileResult {
    std::vector<double> latencies_us;
    uint32_t num_queries = 0;
    double   p50_us = 0.0;
    double   p90_us = 0.0;
    double   p95_us = 0.0;
    double   p99_us = 0.0;
};

/**
 * @brief Distribution of edge distances adopted by the router.
 *
 * An "adopted" edge is one the search actually traverses as its next
 * step — the greedy cursor advance @c (u, v) at upper layers, or an
 * edge whose head neighbor is accepted (@c try_push returns true) into
 * the beam queue at L0.
 *
 * @c edges_by_level   One bucket per layer, indexed by @c layer_id.
 *                     Each entry is the edge distance @c d(u, v)
 *                     between the endpoints of the adopted edge.
 * @c num_queries      Number of queries profiled.
 */
struct EdgeLengthProfileResult {
    std::vector<std::vector<double>> edges_by_level;
    uint32_t num_queries = 0;
};

}   // namespace cpu
}   // namespace artea
