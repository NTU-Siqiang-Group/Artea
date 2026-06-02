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
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Data structure of the artea_graph index. Extends
 *               stacked_rgraph::IndexStructure with the two configs
 *               (propagate / pruning) consumed by per-layer refinement.
 */

#pragma once

#include <artea/cpu/index/stacked_rgraph/index_structure.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/**
 * @brief artea_graph index. Inherits the dynamic hierarchical r-net
 *        backbone from @c stacked_rgraph::IndexStructure and stores the
 *        @c PropagateConfig / @c PruningConfig used when refining a
 *        single layer of the hierarchy as a conv_graph.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure : public stacked_rgraph::IndexStructure<IndexTraitsT> {

    using base_t              = stacked_rgraph::IndexStructure<IndexTraitsT>;
    using vertex_num_t        = typename IndexTraitsT::vertex_num_t;
    using layer_config_t      = typename IndexTraitsT::layer_config_t;
    using rgraph_config_t     = typename IndexTraitsT::artea_graph::rgraph_config_t;
    using propagate_config_t  = typename IndexTraitsT::artea_graph::propagate_config_t;
    using pruning_config_t    = typename IndexTraitsT::artea_graph::pruning_config_t;

    /** @brief Refining capacity per vertex is fixed at 1.5× the rgraph
     *         per-layer @c max_nbr_size. */
    static constexpr auto _refining_max_nbr_size(vertex_num_t rgraph_max_nbr_size)
        -> vertex_num_t
    {
        return static_cast<vertex_num_t>(
            static_cast<double>(rgraph_max_nbr_size) * 1.5);
    }

public:
    /**
     * @brief Construct an empty artea_graph index.
     *
     * @param total_vertices    Expected eventual size of the base dataset
     *                          (forwarded to the stacked_rgraph parent).
     * @param rgraph_config     Stacked r-net geometry / queue / capacity,
     *                          including independent @c ul_max_nbr_size
     *                          and @c bl_max_nbr_size. The two refining
     *                          LayerConfigs are derived as
     *                          @c rgraph_config.{ul,bl}_max_nbr_size() ×
     *                          1.5.
     * @param propagate_config  Conv-graph propagate config used when
     *                          @c refine_layer is invoked.
     * @param pruning_config    artea_graph PruningConfig. Carries RNG
     *                          scale/shift (both consumed by per-layer
     *                          refinement; insertion reads scale_coeffs
     *                          only).
     */
    IndexStructure(
        const vertex_num_t        total_vertices,
        const rgraph_config_t&    rgraph_config,
        const propagate_config_t  propagate_config,
        const pruning_config_t    pruning_config
    ) :
        // Base owns a scale/shift-only PruningConfig (conv_graph shape) —
        // that's what the stacked-rgraph insertion path reads. We project
        // the richer artea_graph config down to that view for the base,
        // and keep the full config locally for refine_layer's ARC sweep.
        base_t(total_vertices, rgraph_config, pruning_config.to_rng_only()),
        _ul_refining_layer_config(
            _refining_max_nbr_size(rgraph_config.ul_max_nbr_size())),
        _bl_refining_layer_config(
            _refining_max_nbr_size(rgraph_config.bl_max_nbr_size())),
        _propagate_config(propagate_config),
        _pruning_config(pruning_config)
    {}

    IndexStructure(const IndexStructure&)            = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&)                 = delete;
    IndexStructure& operator=(IndexStructure&&)      = delete;

    // =================================================================
    //   Config accessors (refinement-time configs)
    // =================================================================

    /** @brief Layer config for refining upper layers (level_id > 0). */
    __attribute__((always_inline))
    auto ul_refining_layer_config() const -> const layer_config_t& {
        return _ul_refining_layer_config;
    }

    __attribute__((always_inline))
    auto ul_refining_layer_config() -> layer_config_t& {
        return _ul_refining_layer_config;
    }

    /** @brief Layer config for refining the bottom layer (L0). */
    __attribute__((always_inline))
    auto bl_refining_layer_config() const -> const layer_config_t& {
        return _bl_refining_layer_config;
    }

    __attribute__((always_inline))
    auto bl_refining_layer_config() -> layer_config_t& {
        return _bl_refining_layer_config;
    }

    /** @brief Refining layer config for @p level_id (bl for L0, ul elsewhere). */
    __attribute__((always_inline))
    auto refining_layer_config(const typename IndexTraitsT::layer_id_t level_id)
        -> layer_config_t&
    {
        return (level_id == 0) ? _bl_refining_layer_config
                               : _ul_refining_layer_config;
    }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& {
        return _propagate_config;
    }

    /** @brief Override the base's scale/shift-only accessor: return the
     *         richer artea_graph PruningConfig (scale + shift + ARC
     *         policy) stored in this derived class. Hides the base
     *         method via name lookup. The base's conv-shape config is
     *         still accessible internally via @c base_t::pruning_config
     *         for the stacked-rgraph insertion path. */
    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& {
        return _pruning_config;
    }

private:
    /** @brief Layer config used to size the RefiningGraph built for
     *         upper layers (level_id > 0) inside @c refine_layer. */
    layer_config_t     _ul_refining_layer_config;

    /** @brief Layer config used to size the RefiningGraph built for
     *         the bottom layer (L0) inside @c refine_layer. */
    layer_config_t     _bl_refining_layer_config;

    /** @brief Conv-graph propagate config for per-layer refinement. */
    propagate_config_t _propagate_config;

    /** @brief Full artea_graph PruningConfig (scale/shift + ARC).
     *         Consumed by @c refine_layer. The base owns a separate
     *         scale/shift-only copy for the insertion path. */
    pruning_config_t   _pruning_config;

};  // class IndexStructure

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
