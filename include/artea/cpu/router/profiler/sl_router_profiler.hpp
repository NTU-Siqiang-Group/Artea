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
 * @FilePath: /Artea/include/artea/cpu/router/profiler/sl_router_profiler.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Per-query 1-NN routing profiler for a single flat layer.
 *               Beam-search from a fixed entry point with a candidate
 *               queue of size @c candidate_queue_size (default 1,
 *               i.e. greedy). Records one
 *               (cumulative NDC, d(q, best)/d(q, true_nn)) point
 *               whenever @c best_dist improves. Returns raw trajectories
 *               — no aggregation.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/profiler/profile_1nn_result.hpp>
#include <artea/cpu/router/detail/make_flat_range.hpp>
#include <artea/cpu/router/detail/make_layer_range.hpp>
#include <artea/cpu/router/data_structures/linear_candidate_queue.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Standalone per-query 1-NN profiler for a single flat graph
 *        (L0-only baseline).
 *
 * For each sampled query, runs a beam search from @p entry_point over
 * the provided graph with a candidate queue of size
 * @c candidate_queue_size. Records, whenever @c best_dist improves, the
 * cumulative number of distance computations (NDC) and the ratio
 * @c best_dist / true_nn_dist. Per-query trajectories are returned raw;
 * queries with @c d(q, true_nn) == 0 are dropped and left as empty
 * trajectories.
 *
 * @c candidate_queue_size == 1 collapses the beam to a single-cursor
 * best-improvement walk (classic greedy).
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class SLRouterProfiler :
    public RouterTraitsT::template vector_router_t<SLRouterProfiler<RouterTraitsT>>
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
        typename RouterTraitsT::template vector_router_t<SLRouterProfiler<RouterTraitsT>>;

public:
    SLRouterProfiler(
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
     * @brief Run the per-query 1-NN profile over a flat single-layer graph.
     *
     * @param query_vecs          Query set.
     * @param single_layer_graph  Flat graph to search.
     * @param gt_vecs             Ground-truth id list; column 0 is the
     *                            top-1 true NN for each query.
     * @param entry_point         Fixed seed vid, same for every query.
     * @param query_indices       Indices into @p query_vecs / @p gt_vecs
     *                            to profile.
     */
    template <typename SingleLayerGraphT>
    auto profile_adr_vs_ndc(
        const query_vecs_t&              query_vecs,
        const SingleLayerGraphT&         single_layer_graph,
        const ground_truth_t&            gt_vecs,
        const vertex_id_t                entry_point,
        const std::vector<vertex_num_t>& query_indices
    ) const -> Profile1NNResult {
        return _profile_adr_vs_ndc_impl(query_vecs, detail::make_flat_range(single_layer_graph), gt_vecs, entry_point, query_indices);
    }

    /**
     * @brief Run the per-query 1-NN profile over a single level of a
     *        hierarchical graph (e.g. L0 of an artea_graph). Cursor is
     *        pinned to @p level_id; no cross-layer descent.
     */
    template <typename HierarchicalGraphT>
    auto profile_adr_vs_ndc(
        const query_vecs_t&              query_vecs,
        const HierarchicalGraphT&        hier_graph,
        const layer_id_t                 level_id,
        const ground_truth_t&            gt_vecs,
        const vertex_id_t                entry_point,
        const std::vector<vertex_num_t>& query_indices
    ) const -> Profile1NNResult {
        return _profile_adr_vs_ndc_impl(query_vecs, detail::make_layer_range(hier_graph, level_id), gt_vecs, entry_point, query_indices);
    }

    /**
     * @brief Measure per-query wall-clock latency over the full query
     *        set. Same beam as @c profile_adr_vs_ndc but without
     *        trajectory bookkeeping, run serially for clean timings.
     */
    template <typename SingleLayerGraphT>
    auto profile_latency(
        const query_vecs_t&      query_vecs,
        const SingleLayerGraphT& single_layer_graph,
        const vertex_id_t        entry_point
    ) const -> LatencyProfileResult {
        return _profile_latency_impl</*RandomSeeding=*/false>(
            query_vecs, detail::make_flat_range(single_layer_graph), entry_point);
    }

    template <typename HierarchicalGraphT>
    auto profile_latency(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph,
        const layer_id_t          level_id,
        const vertex_id_t         entry_point
    ) const -> LatencyProfileResult {
        return _profile_latency_impl</*RandomSeeding=*/false>(
            query_vecs, detail::make_layer_range(hier_graph, level_id), entry_point);
    }

    /**
     * @brief Variant of @c profile_latency that seeds each query with a
     *        fresh random vid sampled uniformly from @c [0, num_vertices).
     *        Meant for the L0-only baseline where "what's the expected
     *        latency from an arbitrary starting point?" is the relevant
     *        question. @p random_seed makes the entry sequence
     *        reproducible across runs.
     */
    template <typename SingleLayerGraphT>
    auto profile_latency_random_entry(
        const query_vecs_t&      query_vecs,
        const SingleLayerGraphT& single_layer_graph,
        const std::uint32_t      random_seed = 42
    ) const -> LatencyProfileResult {
        return _profile_latency_impl</*RandomSeeding=*/true>(
            query_vecs, detail::make_flat_range(single_layer_graph),
            /*entry_point=*/vertex_id_t{0},
            single_layer_graph.get_num_vertices(),
            random_seed);
    }

    // NOTE: @p random_seed is intentionally non-defaulted on this
    //       overload to avoid an ambiguity with the single-layer overload
    //       above — a 3-arg call (q, g, layer_id_t{0}) would otherwise
    //       match both (coercing @c layer_id_t to @c uint32_t for the
    //       flat version's @c random_seed parameter). Callers must pass
    //       a seed explicitly.
    template <typename HierarchicalGraphT>
    auto profile_latency_random_entry(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph,
        const layer_id_t          level_id,
        const std::uint32_t       random_seed
    ) const -> LatencyProfileResult {
        return _profile_latency_impl</*RandomSeeding=*/true>(
            query_vecs, detail::make_layer_range(hier_graph, level_id),
            /*entry_point=*/vertex_id_t{0},
            hier_graph.get_num_vertices(),
            random_seed);
    }

    /**
     * @brief Collect the distance distribution of edges adopted by the
     *        router over the full query set. The single layer the
     *        profiler operates on is bucketed at @c edges_by_level[0].
     *
     * An "adopted" edge is one whose head neighbor is accepted into
     * the beam queue (@c try_push returns @c true) during expansion.
     */
    template <typename SingleLayerGraphT>
    auto profile_edge_length(
        const query_vecs_t&      query_vecs,
        const SingleLayerGraphT& single_layer_graph,
        const vertex_id_t        entry_point
    ) const -> EdgeLengthProfileResult {
        return _profile_edge_length_impl(query_vecs, detail::make_flat_range(single_layer_graph), entry_point);
    }

    template <typename HierarchicalGraphT>
    auto profile_edge_length(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph,
        const layer_id_t          level_id,
        const vertex_id_t         entry_point
    ) const -> EdgeLengthProfileResult {
        return _profile_edge_length_impl(query_vecs, detail::make_layer_range(hier_graph, level_id), entry_point);
    }

private:
    template <typename NeighborRangeT>
    auto _profile_adr_vs_ndc_impl(
        const query_vecs_t&              query_vecs,
        const NeighborRangeT&            nbrs_range,
        const ground_truth_t&            gt_vecs,
        const vertex_id_t                entry_point,
        const std::vector<vertex_num_t>& query_indices
    ) const -> Profile1NNResult {
        const vertex_num_t num_sampled = static_cast<vertex_num_t>(query_indices.size());

        Profile1NNResult result;
        result.trajectories.assign(num_sampled, typename Profile1NNResult::trajectory_t{});
        if (num_sampled == 0) return result;

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

                    const distance_t ep_dist = this->_dist_func(q_vec, this->_vecs_data.get(entry_point));
                    visited.set(entry_point);
                    uint64_t ndc = 1;

                    distance_t best_dist = ep_dist;
                    typename Profile1NNResult::trajectory_t traj;
                    traj.emplace_back(ndc, static_cast<double>(best_dist) * inv_nn_dist);

                    // Beam search with a candidate queue of capacity
                    // _candidate_queue_size. Capacity 1 collapses to
                    // single-cursor greedy best-improvement.
                    candidate_queue_t cq(_candidate_queue_size);
                    cq.try_push(entry_point, ep_dist);

                    while (!cq.empty()) {
                        if (cq.should_terminate()) break;
                        const auto current = cq.pop_best_unexplored_entry();
                        if (current.is_invalid()) break;
                        for (const vertex_id_t nbr_vid : nbrs_range.of(current.get_vid())) {
                            if (visited.test_and_set(nbr_vid)) continue;
                            const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                            ++ndc;
                            if (d < best_dist) {
                                best_dist = d;
                                traj.emplace_back(ndc, static_cast<double>(best_dist) * inv_nn_dist);
                            }
                            cq.try_push(nbr_vid, d);
                        }
                    }

                    result.trajectories[i] = std::move(traj);
                }
            }
        );

        result.num_skipped = local_skipped.combine([](uint32_t a, uint32_t b) { return a + b; });
        return result;
    }

    /**
     * @brief Serial per-query wall-clock latency loop. Uses the same
     *        beam (capacity = @c _candidate_queue_size) as
     *        @c _profile_adr_vs_ndc_impl, but drops trajectory
     *        bookkeeping and runs queries one-by-one to keep timings
     *        free of TBB scheduling noise.
     */
    template <bool RandomSeeding, typename NeighborRangeT>
    auto _profile_latency_impl(
        const query_vecs_t&   query_vecs,
        const NeighborRangeT& nbrs_range,
        const vertex_id_t     entry_point,         // used when !RandomSeeding
        const vertex_num_t    num_vertices = 0,    // used when RandomSeeding
        const std::uint32_t   random_seed  = 42    // used when RandomSeeding
    ) const -> LatencyProfileResult {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();

        LatencyProfileResult result;
        result.latencies_us.assign(num_queries, 0.0);
        result.num_queries = num_queries;
        if (num_queries == 0) return result;

        using clock_t = std::chrono::steady_clock;

        // RNG is declared unconditionally but only consumed in the
        // RandomSeeding path — the branch below is compile-time eliminated.
        // Upper bound is clamped to avoid underflow when num_vertices == 0
        // in the non-random path.
        std::mt19937 rng(random_seed);
        std::uniform_int_distribution<vertex_id_t> ep_sampler(
            vertex_id_t{0},
            num_vertices == 0 ? vertex_id_t{0} : vertex_id_t(num_vertices - 1));

        for (vertex_num_t q = 0; q < num_queries; ++q) {
            const vec_ele_t* q_vec = query_vecs.get(q);

            vertex_id_t ep;
            if constexpr (RandomSeeding) ep = ep_sampler(rng);
            else                         ep = entry_point;

            auto& visited = _visited_table_pool.acquire();

            const auto t0 = clock_t::now();

            const distance_t ep_dist = this->_dist_func(q_vec, this->_vecs_data.get(ep));
            visited.set(ep);

            distance_t best_dist = ep_dist;

            candidate_queue_t cq(_candidate_queue_size);
            cq.try_push(ep, ep_dist);

            while (!cq.empty()) {
                if (cq.should_terminate()) break;
                const auto current = cq.pop_best_unexplored_entry();
                if (current.is_invalid()) break;
                for (const vertex_id_t nbr_vid : nbrs_range.of(current.get_vid())) {
                    if (visited.test_and_set(nbr_vid)) continue;
                    const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                    if (d < best_dist) best_dist = d;
                    cq.try_push(nbr_vid, d);
                }
            }

            const auto t1 = clock_t::now();
            result.latencies_us[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        _fill_percentiles(result);
        return result;
    }

    template <typename NeighborRangeT>
    auto _profile_edge_length_impl(
        const query_vecs_t&   query_vecs,
        const NeighborRangeT& nbrs_range,
        const vertex_id_t     entry_point
    ) const -> EdgeLengthProfileResult {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();

        EdgeLengthProfileResult result;
        result.num_queries = num_queries;
        result.edges_by_level.assign(1, std::vector<double>{});
        if (num_queries == 0) return result;

        tbb::enumerable_thread_specific<std::vector<double>> local_bucket(
            [&]() { return std::vector<double>{}; });

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& bucket = local_bucket.local();
                for (vertex_num_t q = r.begin(); q != r.end(); ++q) {
                    const vec_ele_t* q_vec = query_vecs.get(q);

                    auto& visited = _visited_table_pool.acquire();

                    const distance_t ep_dist = this->_dist_func(q_vec, this->_vecs_data.get(entry_point));
                    visited.set(entry_point);
                    distance_t best_dist = ep_dist;

                    candidate_queue_t cq(_candidate_queue_size);
                    cq.try_push(entry_point, ep_dist);

                    while (!cq.empty()) {
                        if (cq.should_terminate()) break;
                        const auto current = cq.pop_best_unexplored_entry();
                        if (current.is_invalid()) break;
                        const vertex_id_t popped_vid = current.get_vid();
                        const vec_ele_t* popped_vec  = this->_vecs_data.get(popped_vid);
                        for (const vertex_id_t nbr_vid : nbrs_range.of(popped_vid)) {
                            if (visited.test_and_set(nbr_vid)) continue;
                            const distance_t d = this->_dist_func(q_vec, this->_vecs_data.get(nbr_vid));
                            if (d < best_dist) best_dist = d;
                            const bool pushed = cq.try_push(nbr_vid, d);
                            if (pushed) {
                                const distance_t edge_len = this->_dist_func(
                                    popped_vec, this->_vecs_data.get(nbr_vid));
                                bucket.push_back(static_cast<double>(edge_len));
                            }
                        }
                    }
                }
            });

        for (const auto& local : local_bucket) {
            result.edges_by_level[0].insert(
                result.edges_by_level[0].end(),
                local.begin(), local.end());
        }
        return result;
    }

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

    mutable visited_table_pool_t _visited_table_pool;
    std::size_t                  _candidate_queue_size;

};  // class SLRouterProfiler

}   // namespace cpu
}   // namespace artea
