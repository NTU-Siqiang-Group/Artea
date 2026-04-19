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

public:
    /**
     * @brief Construct an empty artea_graph index.
     *
     * @param total_vertices          Expected eventual size of the base dataset
     *                                (forwarded to the stacked_rgraph parent).
     * @param rgraph_config           Stacked r-net geometry / queue / capacity.
     * @param refining_layer_config   Layer config (max_nbr_size / reserved_nbr_size)
     *                                used when constructing the @c RefiningGraph
     *                                for a single layer inside @c refine_layer.
     * @param propagate_config        Conv-graph propagate config used when
     *                                @c refine_layer is invoked.
     * @param pruning_config          Conv-graph pruning config used when
     *                                @c refine_layer is invoked.
     */
    IndexStructure(
        const vertex_num_t        total_vertices,
        const rgraph_config_t&    rgraph_config,
        const layer_config_t      refining_layer_config,
        const propagate_config_t  propagate_config,
        const pruning_config_t    pruning_config
    ) :
        // Stacked-rgraph backbone no longer carries pruning_config — the
        // build-time hierarchical pruner uses plain RNG. pruning_config
        // is kept here because the per-layer refinement pipeline still
        // consumes scale/shift in its PruningUpdater routing loop.
        base_t(total_vertices, rgraph_config),
        _refining_layer_config(refining_layer_config),
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

    __attribute__((always_inline))
    auto refining_layer_config() const -> const layer_config_t& {
        return _refining_layer_config;
    }

    __attribute__((always_inline))
    auto refining_layer_config() -> layer_config_t& {
        return _refining_layer_config;
    }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& {
        return _propagate_config;
    }

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& {
        return _pruning_config;
    }

private:
    /** @brief Layer config used to size the RefiningGraph built per layer
     *         inside @c refine_layer. */
    layer_config_t     _refining_layer_config;

    /** @brief Conv-graph propagate config for per-layer refinement. */
    propagate_config_t _propagate_config;

    /** @brief Conv-graph pruning config for per-layer refinement. */
    pruning_config_t   _pruning_config;

};  // class IndexStructure

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
