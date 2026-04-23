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
 * @FilePath: /Artea/include/artea/cpu/router/profiler/hg_router_profiler.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Per-hop 1-NN routing profiler for the full multi-level
 *               hierarchy. Extracted from
 *               @c HierarchicalGraphRouter::profile_1nn_query so the
 *               profiler lives as a standalone component: greedy descent
 *               with a unified hop axis across layers, records
 *               (cumulative NDC, d(q, best)/d(q, true_nn)) per successful
 *               cursor move, aggregates with sticky-last-value.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/profiler/profile_1nn_result.hpp>
#include <artea/cpu/router/detail/make_layer_range.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Standalone per-hop 1-NN profiler for multi-level hierarchies.
 *
 * Greedy descent from a fixed @p entry_point at the top level down to L0
 * with ONE unified hop axis (hops are not reset at layer boundaries). The
 * visited table is cleared between layers because each layer's
 * neighborhood is different; the cursor @c (best_vid, best_dist) carries
 * across. No L0 beam — pure best-improvement greedy at every level.
 *
 * Aggregation: sticky-last-value on terminated trajectories;
 * @c d(q, true_nn) == 0 queries are dropped and counted separately in
 * @c skipped_queries.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class HGRouterProfiler :
    public RouterTraitsT::template vector_router_t<HGRouterProfiler<RouterTraitsT>>
{
    using vertex_num_t         = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t          = typename RouterTraitsT::vertex_id_t;
    using layer_id_t           = typename RouterTraitsT::layer_id_t;
    using vec_ele_t            = typename RouterTraitsT::vec_ele_t;
    using distance_t           = typename RouterTraitsT::distance_t;
    using dist_func_t          = typename RouterTraitsT::dist_func_t;
    using vector_array_t       = typename RouterTraitsT::vector_array_t;
    using query_vecs_t         = typename RouterTraitsT::query_vecs_t;
    using visited_table_pool_t = typename RouterTraitsT::visited_table_pool_t;
    using ground_truth_t       = typename RouterTraitsT::ground_truth_t;
    using base_class_t         =
        typename RouterTraitsT::template vector_router_t<HGRouterProfiler<RouterTraitsT>>;

public:
    HGRouterProfiler(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func
    ) :
        base_class_t(base_vecs, dist_func, /*topk=*/0),
        _visited_table_pool(base_vecs.get_num_vecs())
    {}

    /** @brief Warm up the visited-table pool so the first profiled query
     *         doesn't pay the per-worker init cost. */
    auto initialize() -> void { _visited_table_pool.warmup(); }

    /**
     * @brief Run the per-hop 1-NN profile across a query set.
     *
     * @param query_vecs   Query set.
     * @param hier_graph   Hierarchical graph (compact or dynamic — selected
     *                     via @c detail::make_layer_range).
     * @param gt_vecs      Ground-truth id list; column 0 is the top-1
     *                     true NN for each query.
     * @param entry_point  Fixed seed vid at the apex, same for every query.
     */
    template <typename HierarchicalGraphT>
    auto profile(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph,
        const ground_truth_t&     gt_vecs,
        const vertex_id_t         entry_point
    ) const -> Profile1NNResult {
        const vertex_num_t num_queries  = query_vecs.get_num_vecs();
        const layer_id_t   top_level_id = hier_graph.top_occupied_level_id();

        Profile1NNResult result;
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id ||
            num_queries == 0)
        {
            return result;
        }

        using traj_t = std::vector<std::pair<uint64_t, double>>;
        std::vector<traj_t> trajectories(num_queries);
        tbb::enumerable_thread_specific<uint32_t> local_skipped(uint32_t{0});

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t q = r.begin(); q != r.end(); ++q) {
                    const vec_ele_t* q_vec = query_vecs.get(q);
                    const vertex_id_t nn_vid =
                        static_cast<vertex_id_t>(gt_vecs.get(q)[0]);
                    const distance_t nn_dist =
                        this->_dist_func(q_vec, this->_vecs_data.get(nn_vid));
                    if (nn_dist == distance_t(0)) {
                        ++local_skipped.local();
                        continue;
                    }
                    const double inv_nn_dist =
                        1.0 / static_cast<double>(nn_dist);

                    auto& visited = _visited_table_pool.acquire();

                    // Seed at the apex.
                    vertex_id_t best_vid  = entry_point;
                    distance_t  best_dist = this->_dist_func(
                        q_vec, this->_vecs_data.get(best_vid));
                    visited.set(best_vid);
                    uint64_t ndc = 1;

                    traj_t traj;
                    traj.emplace_back(ndc,
                        static_cast<double>(best_dist) * inv_nn_dist);

                    // Descend top..0 with a unified hop axis.
                    for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                        const auto nbrs_range =
                            detail::make_layer_range(hier_graph, cur_level_id);

                        while (true) {
                            vertex_id_t next_vid  = best_vid;
                            distance_t  next_dist = best_dist;
                            for (const vertex_id_t nbr_vid : nbrs_range.of(best_vid)) {
                                if (visited.test_and_set(nbr_vid)) continue;
                                const distance_t d = this->_dist_func(
                                    q_vec, this->_vecs_data.get(nbr_vid));
                                ++ndc;
                                if (d < next_dist) {
                                    next_dist = d;
                                    next_vid  = nbr_vid;
                                }
                            }
                            if (next_vid == best_vid) break;   // local optimum on this layer
                            best_vid  = next_vid;
                            best_dist = next_dist;
                            traj.emplace_back(ndc,
                                static_cast<double>(best_dist) * inv_nn_dist);
                        }

                        if (cur_level_id == 0) break;
                        // Different neighborhood on the next layer — cursor
                        // carries, but visited is reset. O(1) with VersionTagTable.
                        visited.clear();
                        visited.set(best_vid);
                    }

                    trajectories[q] = std::move(traj);
                }
            }
        );

        result.skipped_queries =
            local_skipped.combine([](uint32_t a, uint32_t b) { return a + b; });
        result.num_queries = num_queries - result.skipped_queries;

        if (result.num_queries == 0) return result;

        std::size_t max_hops = 0;
        for (const auto& traj : trajectories) {
            if (traj.size() > max_hops) max_hops = traj.size();
        }
        result.per_hop_statistics.resize(max_hops);

        const double inv_valid = 1.0 / static_cast<double>(result.num_queries);
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, max_hops),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t h = r.begin(); h != r.end(); ++h) {
                    double sum_ndc = 0.0;
                    double sum_adr = 0.0;
                    for (const auto& traj : trajectories) {
                        if (traj.empty()) continue;
                        const std::size_t idx = std::min(h, traj.size() - 1);
                        sum_ndc += static_cast<double>(traj[idx].first);
                        sum_adr += traj[idx].second;
                    }
                    result.per_hop_statistics[h] = {
                        sum_ndc * inv_valid,
                        sum_adr * inv_valid
                    };
                }
            }
        );

        return result;
    }

private:
    mutable visited_table_pool_t _visited_table_pool;

};  // class HGRouterProfiler

}   // namespace cpu
}   // namespace artea
