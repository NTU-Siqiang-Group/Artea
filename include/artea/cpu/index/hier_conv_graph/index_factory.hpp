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
 * @FilePath: /Artea/include/artea/cpu/index/hier_conv_graph/index_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Factory for the hier_conv_graph index. Assigns layer
 *               membership by uniform random pull-out (L_{h+1} is a
 *               sample_ratio sample of L_h, stopping when the next
 *               sample would dip below IndexTraits::min_layer_cap),
 *               then refines every occupied layer with the same
 *               conv_graph-style triangle + reverse + truncate
 *               pipeline that artea_graph::refine_layer uses — minus
 *               the ARC sweep and the L0-only shift scaling. Edge
 *               seeding happens fresh per layer via random_eg since
 *               every layer starts edgeless after assign_layer.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace hier_conv_graph {

/**
 * @brief Insertion + refinement engine for the hier_conv_graph index.
 *
 * Standalone factory — does NOT inherit from stacked_rgraph's r-net
 * insertion factory because layer membership is set by random
 * sampling, not r-net radius absorption. The orchestrating call,
 * @c add_vertices, runs two phases:
 *   1. Top-down random pull-out: assigns every newly-inserted vid a
 *      highest_level_id (default 0; ~sample_ratio of L_h vids are
 *      bumped to L_{h+1}); then calls @c assign_layer(vid, h) once
 *      per vid.
 *   2. Per-layer refinement: seeds each layer with random edges via
 *      @c random_eg.generate, then runs the conv_graph-style build
 *      loop on the layer's RefiningGraph (dense for L0, sparse for
 *      L1+ via @c collect_layer_vids).
 *
 * @tparam GraphFactoryTraitsT The graph-factory traits type.
 */
template <typename GraphFactoryTraitsT>
class IndexFactory {

    using this_index_t   = typename GraphFactoryTraitsT::hier_conv_graph::index_t;

    using vertex_num_t   = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t    = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_num_t    = typename GraphFactoryTraitsT::layer_num_t;
    using layer_id_t     = typename GraphFactoryTraitsT::layer_id_t;
    using iter_t         = typename GraphFactoryTraitsT::iter_t;
    using ratio_t        = typename GraphFactoryTraitsT::ratio_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using dist_func_t    = typename GraphFactoryTraitsT::dist_func_t;
    using random_seq_nr_t = typename GraphFactoryTraitsT::random_seq_nr_t;

    using refining_graph_t   = typename GraphFactoryTraitsT::dynamic::refining_graph_t;
    using layer_config_t     = typename GraphFactoryTraitsT::layer_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::hier_conv_graph::propagate_config_t;
    using pruning_config_t   = typename GraphFactoryTraitsT::hier_conv_graph::pruning_config_t;

    using propagate_engine_t = typename GraphFactoryTraitsT::propagate_engine_t;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;
    using pruning_updater_t  = typename GraphFactoryTraitsT::pruning_updater_t;
    using reverse_updater_t  = typename GraphFactoryTraitsT::reverse_updater_t;
    using routing_updater_t  = typename GraphFactoryTraitsT::routing_updater_t;
    using truncate_updater_t = typename GraphFactoryTraitsT::truncate_updater_t;
    using random_eg_t        = typename GraphFactoryTraitsT::random_eg_t;
    using refiner_utils_t    = typename GraphFactoryTraitsT::refiner_utils_t;

public:
    /** @brief Wall-clock breakdown returned by @ref add_vertices. */
    struct BuildTime {
        double level_assignment_ms = 0.0;
        double refinement_ms       = 0.0;
        double total_time_ms       = 0.0;
    };

    /**
     * @brief Append a vector batch to @p index and build the
     *        hier_conv_graph end-to-end.
     *
     * @param index      hier_conv_graph index to grow.
     * @param batch_vecs Batch to insert (moved into @p index).
     * @param dist_func  Distance functor (must outlive this call).
     */
    static auto add_vertices(
        this_index_t& index, vector_array_t&& batch_vecs, const dist_func_t& dist_func
    ) -> BuildTime {
        const auto wallclock_start = std::chrono::high_resolution_clock::now();
        const auto elapsed_ms_since = [](const auto& start) -> double {
            const auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start).count();
        };

        const vertex_num_t batch_size = static_cast<vertex_num_t>(batch_vecs.get_num_vecs());
        if (batch_size == 0) {
            return BuildTime{ 0.0, 0.0, elapsed_ms_since(wallclock_start) };
        }

        index.append_vecs(std::move(batch_vecs));
        const vertex_id_t new_vid_start = index.add_vertices(batch_size);
        const vertex_id_t new_vid_end   = static_cast<vertex_id_t>(new_vid_start + batch_size);

        // ---- Phase 1: assign every new vid a highest_level_id via
        //              top-down random pull-out ----
        const auto level_assignment_start = std::chrono::high_resolution_clock::now();
        _assign_random_levels(index, new_vid_start, new_vid_end);
        const double level_assignment_ms = elapsed_ms_since(level_assignment_start);
        ARTEA_INFO(fmt::format(
            "[hier_conv_graph] random level assignment done in {:.2f} ms (batch_size={}, sample_ratio={})",
            level_assignment_ms, batch_size, index.hierarchy_config().sample_ratio()));

        // ---- Phase 2: refine every occupied layer ----
        const auto refinement_start = std::chrono::high_resolution_clock::now();
        const layer_id_t top_level_id = index.top_occupied_level_id();
        for (layer_id_t level_id = 0; level_id <= top_level_id; ++level_id) {
            refine_layer(index, level_id, dist_func, new_vid_start, new_vid_end);
        }
        const double refinement_ms = elapsed_ms_since(refinement_start);
        const double total_time_ms = elapsed_ms_since(wallclock_start);
        ARTEA_INFO(fmt::format(
            "[hier_conv_graph] per-layer refinement done in {:.2f} ms (total add_vertices: {:.2f} ms)",
            refinement_ms, total_time_ms));

        // Per-layer vertex count breakdown (same accounting trick as
        // artea_graph: count(h) = total - sum(bucket(0..h-1).size())).
        const layer_id_t max_level = index.max_restrict_level();
        const vertex_num_t total_vertices = index.get_num_vertices();
        ARTEA_INFO("[hier_conv_graph] per-layer vertex counts:");
        vertex_num_t running = total_vertices;
        for (layer_id_t h = 0; h <= max_level; ++h) {
            const float layer_ratio_pct = total_vertices > 0
                ? 100.0f * static_cast<float>(running) / static_cast<float>(total_vertices)
                : 0.0f;
            ARTEA_INFO(fmt::format(
                "[hier_conv_graph]   level_id={}: {} vertices ({:.2f}% of base)",
                h, running, layer_ratio_pct));
            running -= static_cast<vertex_num_t>(
                index.get_vids_with_highest_level(h).size());
        }

        return BuildTime{ level_assignment_ms, refinement_ms, total_time_ms };
    }

    /**
     * @brief Refine layer @p level_id by running the conv_graph-style
     *        prune + reverse + truncate pipeline on it.
     *
     * Pipeline (mirrors artea_graph::refine_layer minus ARC sweep and
     * minus the L0-only shift scaling):
     *   1. @c hier_graph.collect_layer_vids(level_id) → (l2g, g2l) vid maps.
     *   2. Allocate refining_graph_t — dense ctor for L0, sparse ctor for L1+.
     *   3. fill_refining_graph_from_layer — no-op when layer is empty
     *      (which it always is here, since assign_layer doesn't seed edges).
     *   4. Skip-small-bucket guard: N_local < min_layer_cap → return.
     *   5. random_eg.generate to seed init_nbr_size random edges per row.
     *      Replaces the conv_graph::_build_loop init step; works on
     *      both dense and sparse refining graphs because
     *      @c get_storage_vid handles the local→global translation.
     *   6. Conv-graph build loop with truncate-inside-triu (artea_graph
     *      pattern):
     *        for build_loop in [0, num_build_loops):
     *          for triu_iter in [0, num_triu_iters):
     *            engine.next(triangle).next(truncate)
     *          engine.next(reverse).next(truncate)
     *   7. Routing loops (optional):
     *        for routing_loop in [0, num_routing_loops):
     *          engine.next(routing).next(pruning).next(reverse).next(truncate)
     *   8. writeback_layer_from_refining_graph.
     *
     * @param index           hier_conv_graph index whose layer is refined.
     * @param level_id        Hierarchy layer to refine.
     * @param dist_func       Distance functor (must outlive this call).
     * @param new_vid_start   Lower bound (inclusive) of the newly
     *                        inserted vid window. Unused today (kept in
     *                        the signature to mirror artea_graph for
     *                        future use, e.g. incremental insertion).
     * @param new_vid_end     Upper bound (exclusive) of the same window.
     */
    static auto refine_layer(
        this_index_t&      index,
        const layer_id_t   level_id,
        const dist_func_t& dist_func,
        const vertex_id_t  new_vid_start,
        const vertex_id_t  new_vid_end
    ) -> void {
        (void)new_vid_start;
        (void)new_vid_end;

        auto& hier_graph              = index.get_hierarchical_graph();
        auto& vecs_storage            = index.get_vecs_storage();
        auto& layer_config            = index.refining_layer_config(level_id);
        const auto& pruning_config    = index.pruning_config();
        const auto& propagate_config  = index.propagate_config();
        const vertex_num_t max_nbr_size = layer_config.max_nbr_size();

        ARTEA_INFO(fmt::format(
            "[hier_conv_graph] refine_layer start: level_id={}, max_nbr_size={}, "
            "prefill_ratio={}, num_build_loops={}, num_triu_iters={}, "
            "num_routing_loops={}, routing_topk={}, routing_queue_size={}",
            level_id, max_nbr_size,
            propagate_config.prefill_ratio(),
            propagate_config.num_build_loops(),
            propagate_config.num_triu_iters(),
            propagate_config.num_routing_loops(),
            propagate_config.resolve_routing_topk(max_nbr_size),
            propagate_config.resolve_routing_queue_size(max_nbr_size)));

        // ---- Step 1: build vid maps for the participating set ----
        auto [local_to_global, global_to_local] = hier_graph.collect_layer_vids(level_id);

        // ---- Step 2: allocate the layer's RefiningGraph ----
        std::unique_ptr<refining_graph_t> refining_graph;
        if (local_to_global.empty()) {
            // L0 / identity mode: every global vid is a row in _nbrs_arr.
            refining_graph = std::make_unique<refining_graph_t>(vecs_storage, layer_config);
        } else {
            // L1+ / sparse mode: local↔global maps drive lookups.
            refining_graph = std::make_unique<refining_graph_t>(
                vecs_storage, layer_config,
                std::move(local_to_global), std::move(global_to_local));
        }

        // ---- Step 3: transcribe any existing edges (no-op here — we
        //              never seed edges via insertion). Kept for
        //              symmetry with artea_graph in case a future
        //              hybrid build seeds upper layers externally. ----
        refiner_utils_t::fill_refining_graph_from_layer(hier_graph, *refining_graph, level_id);

        // ---- Step 4: skip buckets that the compactor will trim ----
        const vertex_num_t num_layer_vertices = refining_graph->get_num_vertices();
        if (num_layer_vertices < GraphFactoryTraitsT::min_layer_cap) {
            ARTEA_INFO(fmt::format(
                "[hier_conv_graph] refine_layer skipping level_id={}: "
                "N_local={} < min_layer_cap={}.",
                level_id, num_layer_vertices, GraphFactoryTraitsT::min_layer_cap));
            return;
        }

        // ---- Step 5: seed initial random edges ----
        // After assign_layer the layer holds zero edges. random_eg
        // populates @c init_nbr_size random neighbors per row; it
        // handles both dense and sparse refining graphs (local→global
        // translation via get_storage_vid).
        const vertex_num_t init_nbr_size = static_cast<vertex_num_t>(
            static_cast<double>(max_nbr_size) * propagate_config.prefill_ratio());
        if (init_nbr_size > 0 && init_nbr_size <= num_layer_vertices) {
            random_eg_t random_eg(dist_func);
            random_eg.generate(*refining_graph, init_nbr_size);
        }

        // ---- Step 6: run prune + reverse + truncate on the RG ----
        propagate_engine_t propagate_engine(dist_func);
        propagate_engine.set_graph(*refining_graph);

        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
        auto reverse_updater  = propagate_engine.template make_updater<reverse_updater_t>();
        auto truncate_updater = propagate_engine.template make_updater<truncate_updater_t>();
        const vertex_num_t routing_topk       = propagate_config.resolve_routing_topk(max_nbr_size);
        const vertex_num_t routing_queue_size = propagate_config.resolve_routing_queue_size(max_nbr_size);
        auto routing_updater  = propagate_engine.template make_updater<routing_updater_t>(routing_topk, routing_queue_size);
        auto pruning_updater  = propagate_engine.template make_updater<pruning_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());

        // Truncate-after-each-triu pattern (artea_graph style).
        for (iter_t build_loop = 0; build_loop < propagate_config.num_build_loops(); ++build_loop) {
            for (iter_t triu_iter = 0; triu_iter < propagate_config.num_triu_iters(); ++triu_iter) {
                propagate_engine.next(triangle_updater).next(truncate_updater);
            }
            propagate_engine.next(reverse_updater).next(truncate_updater);
        }

        // ---- Step 7: optional routing loops ----
        for (iter_t routing_loop = 0; routing_loop < propagate_config.num_routing_loops(); ++routing_loop) {
            propagate_engine.next(routing_updater).next(pruning_updater)
                            .next(reverse_updater).next(truncate_updater);
        }

        // ---- Step 8: write refined edges back ----
        refiner_utils_t::writeback_layer_from_refining_graph(hier_graph, *refining_graph, level_id);
    }

private:
    /**
     * @brief Top-down random pull-out: every new vid starts at L_0;
     *        for h = 1, 2, ... sample @c floor(pool_size * sample_ratio)
     *        vids from the L_{h-1} pool and bump them to L_h. Stop
     *        when the next sample would dip below @c min_layer_cap or
     *        the index's @c max_restrict_level cap. After level
     *        assignment, call @c assign_layer once per new vid.
     */
    static auto _assign_random_levels(
        this_index_t& index, const vertex_id_t new_vid_start, const vertex_id_t new_vid_end
    ) -> void {
        const vertex_num_t batch_size = static_cast<vertex_num_t>(new_vid_end - new_vid_start);
        if (batch_size == 0) return;

        const ratio_t      sample_ratio       = index.hierarchy_config().sample_ratio();
        const layer_num_t  max_restrict_level = index.max_restrict_level();
        constexpr vertex_num_t min_layer_cap  = GraphFactoryTraitsT::min_layer_cap;

        // Each newly-inserted vid's eventual highest_level_id. Default
        // 0 (participates only at the bottom layer); bumped upward as
        // the vid survives successive pull-out rounds.
        std::vector<layer_id_t> highest_level_per_local_vid(batch_size, layer_id_t{0});

        // Pool of vid offsets currently "alive" at the layer being
        // sampled from. Offsets are relative to new_vid_start so they
        // index into highest_level_per_local_vid directly.
        std::vector<vertex_num_t> current_layer_pool(batch_size);
        for (vertex_num_t i = 0; i < batch_size; ++i) {
            current_layer_pool[i] = i;
        }

        random_seq_nr_t random_sampler;
        for (layer_id_t target_layer = layer_id_t{1}; target_layer <= max_restrict_level; ++target_layer) {
            const vertex_num_t current_pool_size = static_cast<vertex_num_t>(current_layer_pool.size());
            const vertex_num_t next_layer_size   = static_cast<vertex_num_t>(
                std::floor(static_cast<double>(current_pool_size) * static_cast<double>(sample_ratio)));
            if (next_layer_size < min_layer_cap) {
                ARTEA_INFO(fmt::format(
                    "[hier_conv_graph] stop pull-out at level_id={}: next_layer_size={} < min_layer_cap={}",
                    target_layer, next_layer_size, min_layer_cap));
                break;
            }

            // RandomSeqNR returns a span backed by an internal scratch
            // buffer that gets overwritten on the next .generate() call,
            // so we copy out the sampled positions before mutating
            // current_layer_pool.
            auto sampled_positions_span = random_sampler.generate(next_layer_size, current_pool_size);
            std::vector<vertex_num_t> next_layer_pool;
            next_layer_pool.reserve(next_layer_size);
            for (const auto position : sampled_positions_span) {
                const vertex_num_t local_vid_offset = current_layer_pool[position];
                highest_level_per_local_vid[local_vid_offset] = target_layer;
                next_layer_pool.push_back(local_vid_offset);
            }
            current_layer_pool = std::move(next_layer_pool);
        }

        // Commit the level decisions to the HierarchicalGraph. Calls
        // are independent across vids; assign_layer's per-arena
        // thread-local slot claim makes parallel calls safe, but doing
        // it serially here keeps the diff minimal — assign_layer is
        // O(1) per vid and the loop is dwarfed by refinement cost.
        for (vertex_num_t i = 0; i < batch_size; ++i) {
            const vertex_id_t new_vid = static_cast<vertex_id_t>(new_vid_start + i);
            index.assign_layer(new_vid, highest_level_per_local_vid[i]);
        }
    }

};  // class IndexFactory

}   // namespace hier_conv_graph
}   // namespace cpu
}   // namespace artea
