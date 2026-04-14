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
 *               and adds @c refine_layer: extract one layer of the
 *               hierarchical graph as a RefiningGraph, run a conv_graph-
 *               style prune + reverse + truncate pipeline on it, then
 *               write the refined edges back.
 */

#pragma once

#include <memory>
#include <utility>

#include <artea/cpu/index/stacked_rgraph/index_factory.hpp>
#include <artea/cpu/index/conv_graph/index_factory.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/**
 * @brief Insertion + refinement engine for the artea_graph index.
 *
 * Inherits @c stacked_rgraph::IndexFactory so callers get the dynamic
 * r-net insertion algorithm. Adds @c refine_layer for driving a
 * conv_graph-style build pass over a single hierarchy layer.
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
    using vector_array_t  = typename GraphFactoryTraitsT::vector_array_t;
    using dist_func_t     = typename GraphFactoryTraitsT::dist_func_t;

    using refining_graph_t = typename GraphFactoryTraitsT::dynamic::refining_graph_t;
    using layer_config_t   = typename GraphFactoryTraitsT::layer_config_t;

    // artea_graph / stacked_rgraph / conv_graph PruningConfig all alias
    // to the same underlying type — one alias is enough.
    using pruning_config_t   = typename GraphFactoryTraitsT::artea_graph::pruning_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::artea_graph::propagate_config_t;

    // Refiner primitives driven directly (conv_graph::construct_graph
    // assumes a dense RefiningGraph, which would break the L1+ sparse
    // case — so we inline the prune + reverse + truncate pipeline here
    // using the same refiner building blocks).
    using propagate_engine_t = typename GraphFactoryTraitsT::propagate_engine_t;
    using pruning_updater_t  = typename GraphFactoryTraitsT::pruning_updater_t;
    using reverse_updater_t  = typename GraphFactoryTraitsT::reverse_updater_t;
    using truncate_updater_t = typename GraphFactoryTraitsT::truncate_updater_t;

public:
    /**
     * @brief Append a vector batch to the index and insert each as a new
     *        vertex via the inherited stacked_rgraph insertion engine.
     *
     * Only @p ul_pruning_config is consumed here — it drives r-net
     * neighbor pruning during upper-layer insertion. The remaining
     * configs (@p bl_pruning_config, @p refining_layer_config,
     * @p refining_propagate_config) are accepted to keep the
     * add-and-refine API symmetric, but actual refinement is triggered
     * separately via @c refine_layer, which reads the stored configs
     * from @p index. Callers that want per-batch override semantics
     * should set @p index's stored configs through construction and
     * pass matching values here.
     *
     * @param index                      The artea_graph index to grow.
     * @param batch_vecs                 Batch to insert (moved into @p index).
     * @param dist_func                  Distance functor (must outlive this call).
     * @param ul_pruning_config          Pruning for upper-layer (stacked_rgraph) insertion.
     * @param bl_pruning_config          Pruning for bottom-layer (conv_graph) refinement.
     * @param refining_layer_config      Layer config used during layer refinement.
     * @param refining_propagate_config  Propagate config used during layer refinement.
     */
    static auto add_vertices(
        this_index_t&             index,
        vector_array_t&&          batch_vecs,
        const dist_func_t&        dist_func,
        const pruning_config_t&   ul_pruning_config,
        [[maybe_unused]] const pruning_config_t&   bl_pruning_config,
        [[maybe_unused]] const layer_config_t&     refining_layer_config,
        [[maybe_unused]] const propagate_config_t& refining_propagate_config
    ) -> void {
        base_t::add_vertices(index, std::move(batch_vecs), dist_func, ul_pruning_config);
    }

    /**
     * @brief Refine layer @p level_id of @p index by running a
     *        conv_graph-style prune + reverse + truncate pipeline over
     *        it.
     *
     * Pipeline:
     *   1. @c hg.collect_layer_vids(level_id) — (l2g, g2l) vid maps.
     *      Empty for @p level_id == 0 (identity mode).
     *   2. Construct a @c refining_graph_t matching the mode: dense
     *      ctor for L0, sparse ctor for L1+. Fill it from the current
     *      layer slot via @c hg.fill_refining_graph_from_layer.
     *   3. Drive @c propagate_engine with @c pruning_updater +
     *      @c reverse_updater + @c truncate_updater directly on the
     *      RefiningGraph. We cannot call
     *      @c conv_factory_t::construct_graph here because its
     *      @c refining_graph_t&& overload dense-constructs its own inner
     *      RefiningGraph and then move-assigns only the nbrs array —
     *      losing the sparse vid mapping.
     *   4. @c hg.writeback_layer_from_refining_graph to push refined
     *      edges back.
     *
     * Refinement configs come from @p index's stored values
     * (set at construction via the IndexStructure ctor).
     *
     * @param index     The artea_graph index whose layer is being refined.
     * @param level_id  Hierarchy layer to refine.
     */
    static auto refine_layer(
        this_index_t&    index,
        const layer_id_t level_id,
        const dist_func_t& dist_func
    ) -> void {
        auto& hg = index.get_hierarchical_graph();
        auto& vecs_storage = index.get_vecs_storage();
        const auto  layer_config    = index.refining_layer_config();
        const auto& pruning_config  = index.pruning_config();

        // ---- Step 1: build vid maps for the participating set ----
        auto [local_to_global, global_to_local] =
            hg.collect_layer_vids(level_id);

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
                std::move(local_to_global),
                std::move(global_to_local));
        }
        hg.fill_refining_graph_from_layer(*refining_graph, level_id);

        // ---- Step 3: run prune + reverse + truncate on the RG ----
        // The log_table inside propagate_engine is indexed by local_vid
        // (N_local for sparse upper layers, N_global in identity mode);
        // set_graph sizes it from the bound RG.
        propagate_engine_t propagate_engine(dist_func);
        propagate_engine.set_graph(*refining_graph);

        auto pruning_updater  = propagate_engine.template make_updater<pruning_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
        auto reverse_updater  = propagate_engine.template make_updater<reverse_updater_t>();
        auto truncate_updater = propagate_engine.template make_updater<truncate_updater_t>();

        propagate_engine.next(pruning_updater)
                        .next(reverse_updater)
                        .next(truncate_updater);

        // ---- Step 4: write refined edges back ----
        hg.writeback_layer_from_refining_graph(*refining_graph, level_id);
    }

};  // class IndexFactory

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
