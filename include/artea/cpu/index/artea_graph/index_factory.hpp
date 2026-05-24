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
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/index_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Factory for the artea_graph index. Inherits the dynamic
 *               r-net insertion logic from stacked_rgraph::IndexFactory
 *               (including L0 neighbors) and adds @c refine_layer:
 *               transcribe each layer of the hierarchical graph into a
 *               RefiningGraph, run a conv_graph-style prune + reverse +
 *               truncate pipeline on it, then write the refined edges
 *               back. The public @c add_vertices (1) inserts the batch
 *               via the parent r-net insertion engine and then (2)
 *               refines every occupied layer in place.
 */

#pragma once

#include <chrono>
#include <memory>
#include <utility>
#include <fmt/format.h>

#include <artea/cpu/index/stacked_rgraph/index_factory.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/**
 * @brief Insertion + refinement engine for the artea_graph index.
 *
 * Inherits @c stacked_rgraph::IndexFactory so the r-net insertion
 * algorithm (now including L0) is reused verbatim. Adds per-layer
 * refinement and orchestrates the full build pipeline in
 * @c add_vertices.
 *
 * @tparam GraphFactoryTraitsT The graph-factory traits type.
 */
template <typename GraphFactoryTraitsT>
class IndexFactory : public stacked_rgraph::IndexFactory<GraphFactoryTraitsT> {

    using base_t          = stacked_rgraph::IndexFactory<GraphFactoryTraitsT>;
    using this_index_t    = typename GraphFactoryTraitsT::artea_graph::index_t;

    using vertex_num_t    = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t     = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_id_t      = typename GraphFactoryTraitsT::layer_id_t;
    using iter_t          = typename GraphFactoryTraitsT::iter_t;
    using ratio_t         = typename GraphFactoryTraitsT::ratio_t;
    using distance_t      = typename GraphFactoryTraitsT::distance_t;
    using vector_array_t  = typename GraphFactoryTraitsT::vector_array_t;
    // dist_func type is per-method template arg (DistFuncT); deduced from caller.

    using refining_graph_t   = typename GraphFactoryTraitsT::dynamic::refining_graph_t;
    using layer_config_t     = typename GraphFactoryTraitsT::layer_config_t;

    // artea_graph / stacked_rgraph / conv_graph PruningConfig all alias
    // to the same underlying type — one alias is enough.
    using pruning_config_t   = typename GraphFactoryTraitsT::artea_graph::pruning_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::artea_graph::propagate_config_t;

    // Refiner primitives driven directly. Each carries a DistFuncT
    // template param, so the trait aliases are also templates here.
    template <typename DistFuncT> using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<DistFuncT>;
    template <typename DistFuncT> using triangle_updater_t = typename GraphFactoryTraitsT::template triangle_updater_t<DistFuncT>;
    template <typename DistFuncT> using pruning_updater_t  = typename GraphFactoryTraitsT::template pruning_updater_t<DistFuncT>;
    template <typename DistFuncT> using reverse_updater_t  = typename GraphFactoryTraitsT::template reverse_updater_t<DistFuncT>;
    template <typename DistFuncT> using routing_updater_t  = typename GraphFactoryTraitsT::template routing_updater_t<DistFuncT>;
    template <typename DistFuncT> using random_updater_t   = typename GraphFactoryTraitsT::template random_updater_t<DistFuncT>;
    template <typename DistFuncT> using truncate_updater_t = typename GraphFactoryTraitsT::template truncate_updater_t<DistFuncT>;
    template <typename DistFuncT> using arc_updater_t      = typename GraphFactoryTraitsT::template arc_updater_t<DistFuncT>;
    using refiner_utils_t    = typename GraphFactoryTraitsT::refiner_utils_t;

public:
    /** @brief Wall-clock breakdown returned by @ref add_vertices. The
     *         two phases match the algorithm's two passes:
     *           - @c upper_layer_time_ms: r-net insertion (the inherited
     *             stacked_rgraph backbone build that produces the upper-
     *             layer hierarchy and, when @c insert_on_L0 == true, the
     *             initial L0 edges).
     *           - @c bottom_layer_time_ms: per-layer refinement loop
     *             (densify + RNG-prune every occupied layer; dominated
     *             by L0 cost since L0 holds the full vertex set).
     *           - @c total_time_ms: end-to-end wall clock of
     *             @ref add_vertices (== upper + bottom modulo trivial
     *             scaffolding around the two phases). */
    struct BuildTime {
        double upper_layer_time_ms  = 0.0;
        double bottom_layer_time_ms = 0.0;
        double total_time_ms        = 0.0;
    };

    /**
     * @brief Append a vector batch to the index and build the artea_graph
     *        end-to-end:
     *          1. Run the inherited r-net insertion engine to grow the
     *             hierarchical backbone. L0 neighbors are built by the
     *             parent factory only when @p insert_on_L0 is true;
     *             otherwise L0 slots are left empty and @c refine_layer
     *             rebuilds them via a random prefill + propagate pipeline.
     *          2. Refine every occupied layer in place by running a
     *             conv_graph-style prune + reverse + truncate pipeline
     *             over it.
     *          3. Write refined edges back into the hierarchical graph.
     *
     * Refinement configs (@c refining_layer_config / @c propagate_config /
     * @c pruning_config) are pulled from @p index (set at construction).
     *
     * @param index                    The artea_graph index to grow.
     * @param batch_vecs               Batch to insert (moved into @p index).
     * @param dist_func                Distance functor (must outlive this call).
     * @param insert_on_L0             Forwarded to the parent r-net factory:
     *                                 when true (default) it builds L0 edges
     *                                 for each new vertex; when false it
     *                                 leaves L0 slots empty. Independently,
     *                                 @c refine_layer always runs a random
     *                                 top-up pass on L0 so that every vertex
     *                                 in the newly-inserted vid window ends
     *                                 up with at least
     *                                 @c refining_max_nbr_size *
     *                                 @c prefill_ratio neighbors before the
     *                                 propagate pipeline starts.
     * @param shuffle_insertion_order  Forwarded to the inherited
     *                                 @c stacked_rgraph::IndexFactory::add_vertices;
     *                                 see that overload for semantics.
     */
    template <typename DistFuncT>
    static auto add_vertices(
        this_index_t&      index,
        vector_array_t&&   batch_vecs,
        const DistFuncT&   dist_func,
        const bool         insert_on_L0 = true,
        const bool         shuffle_insertion_order = false
    ) -> BuildTime {
        // Capture the vid window owned by this batch. refine_layer uses
        // it to scope the L0 random top-up pass to exactly the new
        // vertices — earlier batches' L0 rows stay untouched, and rows
        // that already have enough neighbors (from r-net insertion when
        // insert_on_L0 == true) are skipped via the threshold gate.
        const vertex_id_t new_vid_start =
            static_cast<vertex_id_t>(index.get_num_vertices());

        // Step 1: coarse stacked_rgraph insertion. L0 edge construction
        // is governed by @p insert_on_L0. Either way, refine_layer's
        // random top-up pass will fill L0 rows that still sit below
        // the prefill threshold — no extra orchestration needed here.
        // The base factory already measures wall-clock and returns it
        // via BuildTime; reuse that instead of re-instrumenting here.
        const auto rnet_build_time = base_t::add_vertices(
            index,
            std::move(batch_vecs),
            dist_func,
            /*insert_on_L0=*/insert_on_L0,
            shuffle_insertion_order
        );
        const double upper_layer_time_ms = rnet_build_time.total_time_ms;
        ARTEA_INFO(fmt::format("[artea_graph] r-net insertion done (insert_on_L0={}) in {:.2f} ms",
            insert_on_L0, upper_layer_time_ms));

        const vertex_id_t new_vid_end = static_cast<vertex_id_t>(index.get_num_vertices());

        // Step 2 + 3: refine every occupied layer (including L0) and
        // write back. top_occupied_level_id is 0 for the degenerate
        // single-layer case.
        const auto refine_t0 = std::chrono::high_resolution_clock::now();
        const layer_id_t top_level_id = index.top_occupied_level_id();
        for (layer_id_t level_id = 0; level_id <= top_level_id; ++level_id) {
            refine_layer(
                index, level_id, dist_func,
                new_vid_start, new_vid_end);
        }
        const auto refine_t1 = std::chrono::high_resolution_clock::now();
        const double bottom_layer_time_ms = std::chrono::duration<double, std::milli>(refine_t1 - refine_t0).count();
        const double total_time_ms = upper_layer_time_ms + bottom_layer_time_ms;
        ARTEA_INFO(fmt::format("[artea_graph] per-layer refinement done in {:.2f} ms (total add_vertices: {:.2f} ms)",
            bottom_layer_time_ms, total_time_ms));

        // Per-layer vertex count breakdown. A vertex with
        // highest_level=h' participates in every layer 0..h', so
        // count(h) = total - sum(bucket(0..h-1).size()). Walk bottom-up
        // and decrement `running` by each bucket as we print, no
        // intermediate buffer needed.
        const layer_id_t max_level = index.max_restrict_level();
        const vertex_num_t total_vertices = index.get_num_vertices();
        ARTEA_INFO("[artea_graph] per-layer vertex counts:");
        vertex_num_t running = total_vertices;
        for (layer_id_t h = 0; h <= max_level; ++h) {
            const float ratio = total_vertices > 0
                ? 100.0f * running / static_cast<float>(total_vertices)
                : 0.0f;
            ARTEA_INFO(fmt::format(
                "[artea_graph]   level_id={}: {} vertices ({:.2f}% of base)",
                h, running, ratio));
            running -= static_cast<vertex_num_t>(
                index.get_vids_with_highest_level(h).size());
        }

        return BuildTime{
            upper_layer_time_ms,
            bottom_layer_time_ms,
            total_time_ms
        };
    }

    /**
     * @brief Refine layer @p level_id of @p index by running a
     *        conv_graph-style prune + reverse + truncate pipeline over
     *        it.
     *
     * Pipeline:
     *   1. @c hier_graph.collect_layer_vids(level_id) — (l2g, g2l) vid maps.
     *      Empty for @p level_id == 0 (identity mode).
     *   2. Construct a @c refining_graph_t matching the mode: dense
     *      ctor for L0, sparse ctor for L1+. Fill it from the current
     *      layer slot via @c refiner_utils_t::fill_refining_graph_from_layer.
     *   2.5. (L0 only) run @c random_updater_t in top-up mode over the
     *        newly-inserted vid window: rows whose current neighbor
     *        count is below @c refining_max_nbr_size *
     *        @c prefill_ratio are padded up to that threshold with
     *        random candidates; rows that already meet the threshold
     *        are skipped by the updater itself. This fires regardless
     *        of @c insert_on_L0 — when L0 was built by r-net insertion
     *        the pass is a cheap no-op for rows that already have
     *        enough edges, and it still rescues any sparse rows.
     *   3. Drive @c propagate_engine with @c triangle_updater +
     *      @c reverse_updater + @c truncate_updater (+ optional routing
     *      loops) directly on the RefiningGraph. We cannot call
     *      @c conv_factory_t::construct_graph here because its
     *      @c refining_graph_t&& overload dense-constructs its own inner
     *      RefiningGraph and then move-assigns only the nbrs array —
     *      losing the sparse vid mapping.
     *   4. @c refiner_utils_t::writeback_layer_from_refining_graph to push refined
     *      edges back.
     *
     * All refinement configs come from @p index's stored values (set at
     * construction via the IndexStructure ctor).
     *
     * @param index               The artea_graph index whose layer is being refined.
     * @param level_id            Hierarchy layer to refine.
     * @param dist_func           Distance functor (must outlive this call).
     * @param new_vid_start       Inclusive lower bound of the newly
     *                            inserted vid window (captured in
     *                            @c add_vertices). Used to restrict the
     *                            L0 random top-up pass to exactly the
     *                            new vertices.
     * @param new_vid_end         Exclusive upper bound of the newly
     *                            inserted vid window.
     */
    template <typename DistFuncT>
    static auto refine_layer(
        this_index_t&      index,
        const layer_id_t   level_id,
        const DistFuncT&   dist_func,
        const vertex_id_t  new_vid_start,
        const vertex_id_t  new_vid_end
    ) -> void {
        auto& hier_graph                = index.get_hierarchical_graph();
        auto& vecs_storage              = index.get_vecs_storage();
        // Per-level RefiningGraph layer_config: bl for L0, ul for L1+.
        // Derived in the IndexStructure ctor as rgraph max_nbr_size × 1.5.
        auto& layer_config              = index.refining_layer_config(level_id);
        const auto& pruning_config      = index.pruning_config();
        const auto& propagate_config    = index.propagate_config();
        const vertex_num_t max_nbr_size = layer_config.max_nbr_size();

        ARTEA_INFO(fmt::format(
            "[artea_graph] refine_layer start: level_id={}, max_nbr_size={}, "
            "new_vid_range=[{}, {}), "
            "prefill_ratio={}, num_build_loops={}, num_triu_iters={}, "
            "num_routing_loops={}, routing_topk={}, routing_queue_size={}",
            level_id, max_nbr_size,
            new_vid_start, new_vid_end,
            propagate_config.prefill_ratio(),
            propagate_config.num_build_loops(),
            propagate_config.num_triu_iters(),
            propagate_config.num_routing_loops(),
            propagate_config.resolve_routing_topk(max_nbr_size),
            propagate_config.resolve_routing_queue_size(max_nbr_size)));

        // /** -------------------- Optimization ------------------------------------- ***/
        // /** @brief A sparse graph is efficient enough to search nearest neighbors     */
        // layer_config.max_nbr_size(max_nbr_size / 3);
        // /** ----------------------------------------------------------------------- ***/

        // ---- Step 1: build vid maps for the participating set ----
        auto [local_to_global, global_to_local] = hier_graph.collect_layer_vids(level_id);

        // ---- Step 2: allocate + fill the layer's RefiningGraph ----
        std::unique_ptr<refining_graph_t> refining_graph;
        if (local_to_global.empty()) {
            // L0 / identity mode: dense ctor, _nbrs_arr sized to N_global.
            refining_graph = std::make_unique<refining_graph_t>(vecs_storage, layer_config);
        } else {
            // L1+ / sparse mode: pass moved maps to the sparse ctor.
            refining_graph = std::make_unique<refining_graph_t>(
                vecs_storage, 
                layer_config,
                std::move(local_to_global), 
                std::move(global_to_local)
            );
        }
        // Transcribe the current layer edges (produced by r-net
        // insertion on every layer, L0 included) into the RG.
        refiner_utils_t::fill_refining_graph_from_layer(hier_graph, *refining_graph, level_id);

        // Buckets below @c IndexTraitsT::min_layer_cap get trimmed at
        // compaction (see @c HierarchicalGraphCompactor). Refining them
        // is pure waste, and on a single-vid bucket some updaters rely
        // on non-empty origin_nbrs and would read past the end.
        const vertex_num_t n_local = refining_graph->get_num_vertices();
        if (n_local < GraphFactoryTraitsT::min_layer_cap) {
            ARTEA_INFO(fmt::format(
                "[artea_graph] refine_layer skipping level_id={}: "
                "N_local={} < min_layer_cap={} (bucket will be trimmed "
                "at compaction).",
                level_id, n_local, GraphFactoryTraitsT::min_layer_cap));
            return;
        }

        // ---- Step 3: run prune + reverse + truncate on the RG ----
        // The log_table inside propagate_engine is indexed by local_vid
        // (N_local for sparse upper layers, N_global in identity mode);
        // set_graph sizes it from the bound RG.
        propagate_engine_t<DistFuncT> propagate_engine(dist_func);
        propagate_engine.set_graph(*refining_graph);

        // ---- Step 2.5: L0 random top-up. Fires regardless of how the
        //      r-net insertion step filled L0: RandomUpdater's threshold
        //      gate skips rows that already carry >= prefill_threshold
        //      neighbors and only tops up the deficit on sparse rows.
        //      Scope is restricted to the newly-inserted vid window via
        //      next_range so earlier batches' L0 rows stay untouched.
        if (level_id == 0 && new_vid_end > new_vid_start) {
            const vertex_num_t prefill_threshold = static_cast<vertex_num_t>(
                static_cast<ratio_t>(max_nbr_size) * propagate_config.prefill_ratio());
            if (prefill_threshold > 0) {
                const vertex_num_t l0_num_vertices = refining_graph->get_num_vertices();
                auto random_updater = propagate_engine.template make_updater<random_updater_t<DistFuncT>>(
                        /*rand_gen_size=*/prefill_threshold,
                        /*start_vid=*/vertex_id_t{0},
                        /*end_vid=*/l0_num_vertices);
                propagate_engine.next_range(random_updater, new_vid_start, new_vid_end);
            }
        }

        // R-net covering radius at this layer. Used only by the ARC
        // sweep below to scale its per-layer edge-length cutoff; the
        // shift term consumed by the triangle / pruning updaters is
        // scaled by @c pruning_config.l0_min_distance() instead — see
        // @c effective_shift below.
        const distance_t layer_radius = index.radius_at(level_id);

        // L0-only shift policy: apply @c pruning_config.shifted_coeffs()
        // on the bottom layer only; force shift to 0 at every upper
        // layer. Upper layers' inter-vertex distances are already spread
        // out by the r-net geometry (R_h = R_0 * beta^h), and applying
        // the shift there was empirically over-pruning. Scale coefficient
        // still applies uniformly at every layer; only the shift is gated.
        //
        // The L0 shift is scaled by @c l0_min_distance — a per-dataset
        // characteristic L0 distance, distinct from @c l0_rnet_radius
        // (the r-net L0 covering radius). This lets a single unit-less
        // @c shifted_coeffs grid stay comparable across datasets whose
        // L0 distance scales differ by orders of magnitude. The scaled
        // value is folded into @c effective_shift here so the updaters
        // keep their bare-shift signature.
        const ratio_t effective_shift = (level_id == 0)
            ? pruning_config.shifted_coeffs() * pruning_config.l0_min_distance()
            : ratio_t(0);

        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t<DistFuncT>>(
            pruning_config.scale_coeffs(), effective_shift);
        auto reverse_updater  = propagate_engine.template make_updater<reverse_updater_t<DistFuncT>>();
        const vertex_num_t routing_topk = propagate_config.resolve_routing_topk(max_nbr_size);
        const vertex_num_t routing_queue_size = propagate_config.resolve_routing_queue_size(max_nbr_size);
        auto routing_updater  = propagate_engine.template make_updater<routing_updater_t<DistFuncT>>(routing_topk, routing_queue_size);
        auto truncate_updater = propagate_engine.template make_updater<truncate_updater_t<DistFuncT>>();
        auto pruning_updater = propagate_engine.template make_updater<pruning_updater_t<DistFuncT>>(
            pruning_config.scale_coeffs(), effective_shift);

        // propagate_engine.next(reverse_updater).next(truncate_updater);
        for (iter_t build_loop = 0; build_loop < propagate_config.num_build_loops(); ++build_loop) {
            // propagate_engine.run(propagate_config.num_triu_iters(), triangle_updater).next(truncate_updater)
            //                 .next(reverse_updater).next(truncate_updater);
            for (iter_t triu_iter = 0; triu_iter < propagate_config.num_triu_iters(); ++triu_iter) {
                propagate_engine.next(triangle_updater).next(truncate_updater);
            }
            propagate_engine.next(reverse_updater).next(truncate_updater);
        }

        // /** -------------------- Optimization --------------------------------------- ***/
        // /** @brief Reconstructed as dense graph with original edge number requirements  */
        // layer_config.max_nbr_size(max_nbr_size);
        // /** ------------------------------------------------------------------------- ***/

        for (iter_t routing_loop = 0; routing_loop < propagate_config.num_routing_loops(); ++routing_loop) {
            propagate_engine.next(routing_updater).next(pruning_updater)
                            .next(reverse_updater).next(truncate_updater);
        }

        // ---- Step 3.5: optional aspect-ratio-constrained (ARC) sweep.
        //      Drops every edge longer than
        //      @c aspect_ratio_constraint * @c radius_at(level_id) at
        //      this layer. Applied uniformly across every refined level,
        //      with the threshold scaled per-layer by the r-net covering
        //      radius. Skipped entirely when
        //      @c pruning_config.perform_arc() is false.
        if (pruning_config.perform_arc()) {
            const distance_t arc_threshold = static_cast<distance_t>(
                layer_radius * pruning_config.aspect_ratio_constraint());
            ARTEA_INFO(fmt::format(
                "[artea_graph] refine_layer ARC sweep: level_id={}, "
                "radius={:.6f}, aspect_ratio_constraint={}, arc_threshold={:.6f}",
                level_id, layer_radius,
                pruning_config.aspect_ratio_constraint(), arc_threshold));
            auto arc_updater = propagate_engine.template make_updater<arc_updater_t<DistFuncT>>(arc_threshold);
            propagate_engine.next(arc_updater);
        }

        // ---- Step 4: write refined edges back ----
        refiner_utils_t::writeback_layer_from_refining_graph(hier_graph, *refining_graph, level_id);
    }

};  // class IndexFactory

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
