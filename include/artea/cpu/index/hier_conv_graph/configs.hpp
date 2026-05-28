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
 * @FilePath: /Artea/include/artea/cpu/index/hier_conv_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration types for the hier_conv_graph index.
 *               Layer membership is set by uniform random pull-out
 *               (each upper layer is a @c sample_ratio sample of the
 *               layer below); per-layer edges are then built by the
 *               same conv_graph-style refinement pipeline that
 *               artea_graph uses. Pruning / propagate are reused
 *               verbatim from conv_graph — no ARC, no l0_min_distance.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace hier_conv_graph {

/**
 * @brief Propagate config — aliased straight from conv_graph; the
 *        per-layer refinement loop is identical.
 */
template <typename IndexTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<IndexTraitsT>;

/**
 * @brief Pruning config — aliased straight from conv_graph (scale +
 *        shift, no ARC / l0_min_distance). The hier_conv_graph index
 *        is the "plain hierarchical conv_graph" baseline; artea_graph
 *        is the variant that extends this with ARC pruning.
 */
template <typename IndexTraitsT>
using PruningConfig = conv_graph::PruningConfig<IndexTraitsT>;

/**
 * @brief Hierarchy-shape config for hier_conv_graph.
 *
 * Holds the two per-vertex neighbor capacities (independent ul / bl,
 * mirroring stacked_rgraph::RGraphConfig's split) plus the
 * @c sample_ratio that drives top-down random pull-out: L_{h+1} is a
 * uniformly random @c sample_ratio sample of L_h. There is no rnet
 * beta / L0_radius here — geometry is irrelevant when layer
 * membership is purely random.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
struct HierarchyConfig {
    using ratio_t      = typename IndexTraitsT::ratio_t;
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;

    /**
     * @brief Construct a HierarchyConfig.
     *
     * @param ul_max_nbr_size Per-vertex neighbor capacity at every
     *                        upper layer (level_id > 0).
     * @param bl_max_nbr_size Per-vertex neighbor capacity at the
     *                        bottom layer (level_id == 0). Independent
     *                        of @p ul_max_nbr_size — no hardcoded
     *                        ratio.
     * @param sample_ratio    Fraction of L_h vids promoted to L_{h+1}.
     *                        Default 0.02 (HNSW-like sparse upper
     *                        layers). Must be in (0, 1); typical range
     *                        [0.01, 0.1].
     */
    HierarchyConfig(
        vertex_num_t ul_max_nbr_size,
        vertex_num_t bl_max_nbr_size,
        ratio_t      sample_ratio = ratio_t(0.02)
    ) :
        _ul_max_nbr_size(ul_max_nbr_size),
        _bl_max_nbr_size(bl_max_nbr_size),
        _sample_ratio(sample_ratio)
    {}

    // Builder-pattern setters (chainable)
    auto ul_max_nbr_size(vertex_num_t v) -> HierarchyConfig& { _ul_max_nbr_size = v; return *this; }
    auto bl_max_nbr_size(vertex_num_t v) -> HierarchyConfig& { _bl_max_nbr_size = v; return *this; }
    auto sample_ratio(ratio_t v)         -> HierarchyConfig& { _sample_ratio = v;    return *this; }

    // Const getters
    auto ul_max_nbr_size() const -> vertex_num_t { return _ul_max_nbr_size; }
    auto bl_max_nbr_size() const -> vertex_num_t { return _bl_max_nbr_size; }
    auto sample_ratio()    const -> ratio_t      { return _sample_ratio; }

    /** @brief Per-vertex capacity at @p level_id (bl for L0, ul elsewhere). */
    __attribute__((always_inline))
    auto max_nbr_size(const typename IndexTraitsT::layer_id_t level_id) const
        -> vertex_num_t
    {
        return (level_id == 0) ? _bl_max_nbr_size : _ul_max_nbr_size;
    }

private:
    vertex_num_t _ul_max_nbr_size;
    vertex_num_t _bl_max_nbr_size;
    ratio_t      _sample_ratio;
};

}   // namespace hier_conv_graph
}   // namespace cpu
}   // namespace artea
