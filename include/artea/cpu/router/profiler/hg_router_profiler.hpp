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
 * @Description: Per-query 1-NN routing profiler for the full multi-level
 *               hierarchy. Greedy best-improvement at every upper layer,
 *               beam-search with a configurable candidate queue size at
 *               L0. Records one (cumulative NDC, d(q, best)/d(q, true_nn))
 *               point whenever @c best_dist improves. Returns raw
 *               trajectories — no aggregation.
 */

#pragma once

#include <algorithm>
#include <chrono>
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
#include <artea/cpu/router/data_structures/linear_candidate_queue.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Standalone per-query 1-NN profiler for multi-level hierarchies.
 *
 * Descent from a fixed @p entry_point at the top level down to L0 with
 * ONE unified hop axis (hops are not reset at layer boundaries):
 *   - Upper layers (level > 0): best-improvement greedy walk.
 *   - Bottom layer (L0):        beam search with a candidate queue of
 *                               size @c candidate_queue_size (= 1 gives
 *                               back the greedy baseline).
 * The visited table is cleared between layers because each layer's
 * neighborhood is different; the cursor @c (best_vid, best_dist) carries
 * across and seeds the L0 beam.
 *
 * Each query contributes a raw trajectory; the caller can aggregate or
 * plot them directly. @c d(q, true_nn) == 0 queries are dropped with a
 * warning and their trajectory is left empty.
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
    using candidate_queue_t    = typename RouterTraitsT::candidate_queue_t;
    using base_class_t         =
        typename RouterTraitsT::template vector_router_t<HGRouterProfiler<RouterTraitsT>>;

public:
    HGRouterProfiler(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func,
        const std::size_t     candidate_queue_size = 1
    ) :
        base_class_t(base_vecs, dist_func, /*topk=*/0),
        _visited_table_pool(base_vecs.get_num_vecs()),
        _candidate_queue_size(std::max<std::size_t>(1, candidate_queue_size))
    {}

    /** @brief Warm up the visited-table pool so the first profiled query
     *         doesn't pay the per-worker init cost. */
    auto initialize() -> void { _visited_table_pool.warmup(); }

    /**
     * @brief Run the per-query 1-NN profile over a selected subset of
     *        queries. Upper layers use greedy best-improvement; L0 uses
     *        a beam search with the queue size supplied to the
     *        constructor (@c candidate_queue_size).
     *
     * @param query_vecs     Query set.
     * @param hier_graph     Hierarchical graph (compact or dynamic —
     *                       selected via @c detail::make_layer_range).
     * @param gt_vecs        Ground-truth id list; column 0 is the top-1
     *                       true NN for each query.
     * @param entry_point    Fixed seed vid at the apex, same for every query.
     * @param query_indices  Indices into @p query_vecs / @p gt_vecs to
     *                       profile. The returned trajectories are indexed
     *                       to match this list 1:1.
     */
    template <typename HierarchicalGraphT>
    auto profile_adr_vs_ndc(
        const query_vecs_t&                query_vecs,
        const HierarchicalGraphT&          hier_graph,
        const ground_truth_t&              gt_vecs,
        const vertex_id_t                  entry_point,
        const std::vector<vertex_num_t>&   query_indices
    ) const -> Profile1NNResult {
        const vertex_num_t num_sampled = static_cast<vertex_num_t>(query_indices.size());
        const layer_id_t   top_level_id = hier_graph.top_occupied_level_id();

        Profile1NNResult result;
        result.trajectories.assign(num_sampled, typename Profile1NNResult::trajectory_t{});
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id || num_sampled == 0) {
            return result;
        }

        tbb::enumerable_thread_specific<uint32_t> local_skipped(uint32_t{0});

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_sampled),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_num_t q = query_indices[i];
                    const vec_ele_t* q_vec = query_vecs.get(q);
                    const vertex_id_t nn_vid = static_cast<vertex_id_t>(gt_vecs.get(q)[0]);
                    const distance_t nn_dist = this->_dist_func(q_vec, this->_vecs_data.get(nn_vid));
                    if (nn_dist == distance_t(0)) {
                        ARTEA_WARN(fmt::format("Query {} has zero distance to its ground-truth NN (vid {}); skipping to avoid div-by-zero.", q, nn_vid));
                        ++local_skipped.local();
                        continue;
                    }
                    const double inv_nn_dist = 1.0 / static_cast<double>(nn_dist);
                    auto& visited = _visited_table_pool.acquire();
                    // Seed at the apex.
                    vertex_id_t best_vid  = entry_point;
                    distance_t  best_dist = this->_dist_func(q_vec, this->_vecs_data.get(best_vid));
                    visited.set(best_vid);
                    uint64_t ndc = 1;
                    typename Profile1NNResult::trajectory_t traj;
                    traj.emplace_back(ndc, static_cast<double>(best_dist) * inv_nn_dist);

                    // Descend top..0 with a unified hop axis. Upper
                    // layers: greedy best-improvement. L0: beam search
                    // with a candidate queue seeded by the carried cursor.
                    for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                        const auto nbrs_range = detail::make_layer_range(hier_graph, cur_level_id);

                        if (cur_level_id > 0) {
                            while (true) {
                                vertex_id_t next_vid  = best_vid;
                                distance_t  next_dist = best_dist;
                                for (const vertex_id_t nbr_vid : nbrs_range.of(best_vid)) {
                                    if (visited.test_and_set(nbr_vid)) continue;
                                    const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                                    ++ndc;
                                    if (d < next_dist) {
                                        next_dist = d;
                                        next_vid  = nbr_vid;
                                    }
                                }
                                if (next_vid == best_vid) break;   // local optimum on this layer
                                best_vid  = next_vid;
                                best_dist = next_dist;
                                traj.emplace_back(ndc, static_cast<double>(best_dist) * inv_nn_dist);
                            }
                        } else {
                            // L0 beam search. Seed the queue with the
                            // carried cursor; best_vid is already marked
                            // visited from the L1 transition, so its
                            // neighbors (not itself) are what gets
                            // expanded on the first pop.
                            candidate_queue_t cq(_candidate_queue_size);
                            cq.try_push(best_vid, best_dist);

                            while (!cq.empty()) {
                                if (cq.should_terminate()) break;
                                const auto current = cq.pop_best_unexplored_entry();
                                if (current.is_invalid()) break;
                                for (const vertex_id_t nbr_vid : nbrs_range.of(current.get_vid())) {
                                    if (visited.test_and_set(nbr_vid)) continue;
                                    const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                                    ++ndc;
                                    if (d < best_dist) {
                                        best_vid  = nbr_vid;
                                        best_dist = d;
                                        traj.emplace_back(ndc, static_cast<double>(best_dist) * inv_nn_dist);
                                    }
                                    cq.try_push(nbr_vid, d);
                                }
                            }
                        }

                        if (cur_level_id == 0) break;
                        // Different neighborhood on the next layer — cursor
                        // carries, but visited is reset. O(1) with VersionTagTable.
                        visited.clear();
                        visited.set(best_vid);
                    }

                    result.trajectories[i] = std::move(traj);
                }
            }
        );

        result.num_skipped = local_skipped.combine([](uint32_t a, uint32_t b) { return a + b; });
        return result;
    }

    /**
     * @brief Measure per-query wall-clock latency over the full query
     *        set. Uses the same greedy-upper / L0-beam search as
     *        @c profile_adr_vs_ndc but strips the trajectory bookkeeping
     *        and runs serially (single thread) to keep per-query timing
     *        free of scheduler/contention noise.
     *
     * @param query_vecs    Query set — every query is measured.
     * @param hier_graph    Hierarchical graph.
     * @param entry_point   Fixed apex seed.
     * @return @c LatencyProfileResult with @c latencies_us sized to
     *         @c query_vecs.get_num_vecs() and p50/p90/p95/p99
     *         precomputed over it.
     */
    template <typename HierarchicalGraphT>
    auto profile_latency(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph,
        const vertex_id_t         entry_point
    ) const -> LatencyProfileResult {
        const vertex_num_t num_queries  = query_vecs.get_num_vecs();
        const layer_id_t   top_level_id = hier_graph.top_occupied_level_id();

        LatencyProfileResult result;
        result.latencies_us.assign(num_queries, 0.0);
        result.num_queries = num_queries;
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id || num_queries == 0) {
            return result;
        }

        using clock_t = std::chrono::steady_clock;

        for (vertex_num_t q = 0; q < num_queries; ++q) {
            const vec_ele_t* q_vec = query_vecs.get(q);

            auto& visited = _visited_table_pool.acquire();

            const auto t0 = clock_t::now();

            vertex_id_t best_vid  = entry_point;
            distance_t  best_dist = this->_dist_func(q_vec, this->_vecs_data.get(best_vid));
            visited.set(best_vid);

            for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                const auto nbrs_range = detail::make_layer_range(hier_graph, cur_level_id);

                if (cur_level_id > 0) {
                    while (true) {
                        vertex_id_t next_vid  = best_vid;
                        distance_t  next_dist = best_dist;
                        for (const vertex_id_t nbr_vid : nbrs_range.of(best_vid)) {
                            if (visited.test_and_set(nbr_vid)) continue;
                            const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                            if (d < next_dist) { next_dist = d; next_vid = nbr_vid; }
                        }
                        if (next_vid == best_vid) break;
                        best_vid  = next_vid;
                        best_dist = next_dist;
                    }
                } else {
                    candidate_queue_t cq(_candidate_queue_size);
                    cq.try_push(best_vid, best_dist);
                    while (!cq.empty()) {
                        if (cq.should_terminate()) break;
                        const auto current = cq.pop_best_unexplored_entry();
                        if (current.is_invalid()) break;
                        for (const vertex_id_t nbr_vid : nbrs_range.of(current.get_vid())) {
                            if (visited.test_and_set(nbr_vid)) continue;
                            const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                            if (d < best_dist) { best_vid = nbr_vid; best_dist = d; }
                            cq.try_push(nbr_vid, d);
                        }
                    }
                }

                if (cur_level_id == 0) break;
                visited.clear();
                visited.set(best_vid);
            }

            const auto t1 = clock_t::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.latencies_us[q] = us;
        }

        _fill_percentiles(result);
        return result;
    }

    /**
     * @brief Collect the distance distribution of edges adopted by the
     *        router at each layer over the full query set.
     *
     * An "adopted" edge is one the router traverses as its next step:
     *   - Upper-layer greedy: each cursor advance contributes one edge
     *     @c (prev_best, new_best).
     *   - L0 beam: each neighbor that is accepted (@c try_push returns
     *     @c true) into the beam queue contributes the edge
     *     @c (popped_cursor, neighbor).
     * For every such adopted edge, @c dist(u, v) between its two
     * endpoints (computed via @c _dist_func over @c vecs_data) is
     * appended to the matching layer's bucket.
     *
     * @return @c EdgeLengthProfileResult with @c edges_by_level sized
     *         to @c top_level_id + 1.
     */
    template <typename HierarchicalGraphT>
    auto profile_edge_length(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph,
        const vertex_id_t         entry_point
    ) const -> EdgeLengthProfileResult {
        const vertex_num_t num_queries  = query_vecs.get_num_vecs();
        const layer_id_t   top_level_id = hier_graph.top_occupied_level_id();

        EdgeLengthProfileResult result;
        result.num_queries = num_queries;
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id || num_queries == 0) {
            return result;
        }

        const std::size_t num_layers = static_cast<std::size_t>(top_level_id) + 1;
        result.edges_by_level.assign(num_layers, std::vector<double>{});

        tbb::enumerable_thread_specific<std::vector<std::vector<double>>> local_buckets(
            [&]() { return std::vector<std::vector<double>>(num_layers); });

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& buckets = local_buckets.local();
                for (vertex_num_t q = r.begin(); q != r.end(); ++q) {
                    const vec_ele_t* q_vec = query_vecs.get(q);

                    auto& visited = _visited_table_pool.acquire();

                    vertex_id_t best_vid  = entry_point;
                    distance_t  best_dist = this->_dist_func(q_vec, this->_vecs_data.get(best_vid));
                    visited.set(best_vid);

                    for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                        const auto nbrs_range = detail::make_layer_range(hier_graph, cur_level_id);

                        if (cur_level_id > 0) {
                            while (true) {
                                vertex_id_t next_vid  = best_vid;
                                distance_t  next_dist = best_dist;
                                for (const vertex_id_t nbr_vid : nbrs_range.of(best_vid)) {
                                    if (visited.test_and_set(nbr_vid)) continue;
                                    const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                                    if (d < next_dist) { next_dist = d; next_vid = nbr_vid; }
                                }
                                if (next_vid == best_vid) break;
                                // Adopted edge: (best_vid, next_vid).
                                const distance_t edge_len = this->_dist_func(
                                    this->_vecs_data.get(best_vid), this->_vecs_data.get(next_vid));
                                buckets[cur_level_id].push_back(static_cast<double>(edge_len));
                                best_vid  = next_vid;
                                best_dist = next_dist;
                            }
                        } else {
                            candidate_queue_t cq(_candidate_queue_size);
                            cq.try_push(best_vid, best_dist);
                            while (!cq.empty()) {
                                if (cq.should_terminate()) break;
                                const auto current = cq.pop_best_unexplored_entry();
                                if (current.is_invalid()) break;
                                const vertex_id_t popped_vid = current.get_vid();
                                const vec_ele_t* popped_vec  = this->_vecs_data.get(popped_vid);
                                for (const vertex_id_t nbr_vid : nbrs_range.of(popped_vid)) {
                                    if (visited.test_and_set(nbr_vid)) continue;
                                    const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                                    if (d < best_dist) { best_vid = nbr_vid; best_dist = d; }
                                    const bool pushed = cq.try_push(nbr_vid, d);
                                    if (pushed) {
                                        const distance_t edge_len = this->_dist_func(
                                            popped_vec, this->_vecs_data.get(nbr_vid));
                                        buckets[0].push_back(static_cast<double>(edge_len));
                                    }
                                }
                            }
                        }

                        if (cur_level_id == 0) break;
                        visited.clear();
                        visited.set(best_vid);
                    }
                }
            });

        // Merge thread-local buckets.
        for (const auto& local : local_buckets) {
            for (std::size_t l = 0; l < num_layers; ++l) {
                result.edges_by_level[l].insert(
                    result.edges_by_level[l].end(),
                    local[l].begin(), local[l].end());
            }
        }
        return result;
    }

private:
    mutable visited_table_pool_t _visited_table_pool;
    std::size_t                  _candidate_queue_size;

    static auto _fill_percentiles(LatencyProfileResult& r) -> void {
        if (r.latencies_us.empty()) return;
        std::vector<double> sorted = r.latencies_us;
        std::sort(sorted.begin(), sorted.end());
        const auto pick = [&](double p) -> double {
            const std::size_t n = sorted.size();
            std::size_t idx = static_cast<std::size_t>(static_cast<double>(n) * p);
            if (idx >= n) idx = n - 1;
            return sorted[idx];
        };
        r.p50_us = pick(0.50);
        r.p90_us = pick(0.90);
        r.p95_us = pick(0.95);
        r.p99_us = pick(0.99);
    }

};  // class HGRouterProfiler

}   // namespace cpu
}   // namespace artea
