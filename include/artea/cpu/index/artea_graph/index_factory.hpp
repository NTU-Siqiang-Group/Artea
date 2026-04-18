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

#include <memory>
#include <utility>

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
    using vector_array_t  = typename GraphFactoryTraitsT::vector_array_t;
    using dist_func_t     = typename GraphFactoryTraitsT::dist_func_t;

    using refining_graph_t   = typename GraphFactoryTraitsT::dynamic::refining_graph_t;
    using layer_config_t     = typename GraphFactoryTraitsT::layer_config_t;

    // artea_graph / stacked_rgraph / conv_graph PruningConfig all alias
    // to the same underlying type — one alias is enough.
    using pruning_config_t   = typename GraphFactoryTraitsT::artea_graph::pruning_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::artea_graph::propagate_config_t;

    // Refiner primitives driven directly (conv_graph::construct_graph
    // assumes a dense RefiningGraph, which would break the L1+ sparse
    // case — so we inline the prune + reverse + truncate pipeline here
    // using the same refiner building blocks).
    using propagate_engine_t = typename GraphFactoryTraitsT::propagate_engine_t;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;
    using pruning_updater_t  = typename GraphFactoryTraitsT::pruning_updater_t;
    using reverse_updater_t  = typename GraphFactoryTraitsT::reverse_updater_t;
    using routing_updater_t  = typename GraphFactoryTraitsT::routing_updater_t;
    using truncate_updater_t = typename GraphFactoryTraitsT::truncate_updater_t;
    using refiner_utils_t    = typename GraphFactoryTraitsT::refiner_utils_t;

public:
    /**
     * @brief Append a vector batch to the index and build the artea_graph
     *        end-to-end:
     *          1. Run the inherited r-net insertion engine to grow the
     *             hierarchical backbone, including L0 neighbors.
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
     * @param shuffle_insertion_order  Forwarded to the inherited
     *                                 @c stacked_rgraph::IndexFactory::add_vertices;
     *                                 see that overload for semantics.
     */
    static auto add_vertices(
        this_index_t&      index,
        vector_array_t&&   batch_vecs,
        const dist_func_t& dist_func,
        const bool         shuffle_insertion_order = false
    ) -> void {
        // Step 1: coarse stacked_rgraph insertion across every layer
        // (L0 included). Each layer's slot is seeded with the r-net
        // candidates produced by Phase 1/2 so refine_layer has real
        // edges to prune rather than empty rows.
        base_t::add_vertices(index, std::move(batch_vecs), dist_func,
                             /*insert_on_L0=*/true,
                             shuffle_insertion_order);

        // Step 2 + 3: refine every occupied layer (including L0) and
        // write back. top_occupied_level_id is 0 for the degenerate
        // single-layer case.
        const layer_id_t top_level_id = index.top_occupied_level_id();
        for (layer_id_t level_id = 0; level_id <= top_level_id; ++level_id) {
            refine_layer(index, level_id, dist_func);
        }
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
     * @param index          The artea_graph index whose layer is being refined.
     * @param level_id       Hierarchy layer to refine.
     * @param dist_func      Distance functor (must outlive this call).
     */
    static auto refine_layer(
        this_index_t&      index,
        const layer_id_t   level_id,
        const dist_func_t& dist_func
    ) -> void {
        auto& hier_graph              = index.get_hierarchical_graph();
        auto& vecs_storage            = index.get_vecs_storage();
        auto& layer_config            = index.refining_layer_config();
        const auto& pruning_config    = index.pruning_config();
        const auto& propagate_config  = index.propagate_config();

        const vertex_num_t max_nbr_size = layer_config.max_nbr_size();

        // /** -------------------- Optimization ------------------------------------- ***/
        // /** @brief A sparse graph is efficient enough to search nearest neighbors     */
        // layer_config.max_nbr_size(max_nbr_size / 2);
        // /** ----------------------------------------------------------------------- ***/

        // ---- Step 1: build vid maps for the participating set ----
        auto [local_to_global, global_to_local] = hier_graph.collect_layer_vids(level_id);

        // ---- Step 2: allocate + fill the layer's RefiningGraph ----
        std::unique_ptr<refining_graph_t> refining_graph;
        if (local_to_global.empty()) {
            // L0 / identity mode: dense ctor, _nbrs_arr sized to N_global.
            refining_graph = std::make_unique<refining_graph_t>(
                vecs_storage, layer_config);
        } else {
            // L1+ / sparse mode: pass moved maps to the sparse ctor.
            refining_graph = std::make_unique<refining_graph_t>(
                vecs_storage, layer_config,
                std::move(local_to_global), std::move(global_to_local));
        }
        // Transcribe the current layer edges (produced by r-net
        // insertion on every layer, L0 included) into the RG.
        refiner_utils_t::fill_refining_graph_from_layer(
            hier_graph, *refining_graph, level_id);

        // ---- Step 3: run prune + reverse + truncate on the RG ----
        // The log_table inside propagate_engine is indexed by local_vid
        // (N_local for sparse upper layers, N_global in identity mode);
        // set_graph sizes it from the bound RG.
        propagate_engine_t propagate_engine(dist_func);
        propagate_engine.set_graph(*refining_graph);

        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
        auto reverse_updater  = propagate_engine.template make_updater<reverse_updater_t>();
        const vertex_num_t routing_topk = propagate_config.resolve_routing_topk(max_nbr_size);
        const vertex_num_t routing_queue_size = propagate_config.resolve_routing_queue_size(max_nbr_size);
        auto routing_updater  = propagate_engine.template make_updater<routing_updater_t>(
            routing_topk, routing_queue_size);
        auto truncate_updater = propagate_engine.template make_updater<truncate_updater_t>();

        for (iter_t build_loop = 0;
             build_loop < propagate_config.num_build_loops();
             ++build_loop)
        {
            propagate_engine.next(reverse_updater).next(truncate_updater)
                            .run(propagate_config.num_triu_iters(), triangle_updater);
        }

        // /** -------------------- Optimization --------------------------------------- ***/
        // /** @brief Reconstructed as dense graph with original edge number requirements  */
        // layer_config.max_nbr_size(max_nbr_size);
        // /** ------------------------------------------------------------------------- ***/

        for (iter_t routing_loop = 0;
             routing_loop < propagate_config.num_routing_loops();
             ++routing_loop)
        {
            auto pruning_updater = propagate_engine.template make_updater<pruning_updater_t>(
                pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
            propagate_engine.next(routing_updater).next(pruning_updater)
                            .next(truncate_updater).next(reverse_updater).next(truncate_updater);
        }

        // ---- Step 4: write refined edges back ----
        refiner_utils_t::writeback_layer_from_refining_graph(
            hier_graph, *refining_graph, level_id);
    }

};  // class IndexFactory

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
