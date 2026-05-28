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
 * @FilePath: /Artea/include/artea/cpu/index/hier_conv_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Data structure of the hier_conv_graph index. Composes
 *               a dynamic::HierarchicalGraph (no r-net config — layer
 *               membership is set externally by the factory via random
 *               pull-out) and holds the conv_graph-style propagate /
 *               pruning configs consumed during per-layer refinement.
 *               Mirrors stacked_rgraph::IndexStructure's forwarder
 *               pattern but with a slimmer config set.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include <tbb/concurrent_vector.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/index/hier_conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace hier_conv_graph {

/**
 * @brief hier_conv_graph index. Owns:
 *   - A composed dynamic::HierarchicalGraph (unique_ptr — non-movable).
 *   - The HierarchyConfig (ul / bl capacities + sample_ratio).
 *   - The conv_graph propagate / pruning configs used by per-layer refinement.
 *   - An owned copy of every base vector inserted via append_vecs.
 *   - Two derived LayerConfig instances (1.5x max_nbr_size, mirroring
 *     artea_graph::IndexStructure).
 *
 * Copy / move are both deleted — clients hold an instance inside a
 * std::unique_ptr.
 */
template <typename IndexTraitsT>
class IndexStructure {

public:
    using vertex_num_t         = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t          = typename IndexTraitsT::vertex_id_t;
    using layer_num_t          = typename IndexTraitsT::layer_num_t;
    using layer_id_t           = typename IndexTraitsT::layer_id_t;
    using ratio_t              = typename IndexTraitsT::ratio_t;
    using nbr_t                = typename IndexTraitsT::nbr_t;
    using layer_config_t       = typename IndexTraitsT::layer_config_t;
    using vector_array_t       = typename IndexTraitsT::vector_array_t;
    using vecs_storage_t       = typename IndexTraitsT::vecs_storage_t;
    using hierarchy_config_t   = typename IndexTraitsT::hier_conv_graph::hierarchy_config_t;
    using propagate_config_t   = typename IndexTraitsT::hier_conv_graph::propagate_config_t;
    using pruning_config_t     = typename IndexTraitsT::hier_conv_graph::pruning_config_t;

    using hierarchical_graph_t = typename IndexTraitsT::dynamic::hierarchical_graph_t;

    /** @brief Forwards the dynamic-mode flag so router-side
     *         make_layer_range picks the right NeighborRange adapter. */
    static constexpr bool is_compacted = hierarchical_graph_t::is_compacted;

    static constexpr layer_id_t unassigned_highest_level_id =
        hierarchical_graph_t::unassigned_highest_level_id;

    /** @brief Refining capacity per vertex is 1.5x the layer's
     *         max_nbr_size — same heuristic as artea_graph. */
    static constexpr auto refining_max_nbr_size_for(vertex_num_t max_nbr_size) -> vertex_num_t {
        return static_cast<vertex_num_t>(static_cast<double>(max_nbr_size) * 1.5);
    }

    /**
     * @brief Conservative upper bound on usable layers under uniform
     *        random pull-out with ratio @p sample_ratio.
     *
     * After @c h sampling rounds the expected layer size is
     * @c total_vertices * sample_ratio^h. The factory stops once that
     * size would drop below @c min_layer_cap, so we size
     * @c max_restrict_level one above that threshold — generous
     * enough to absorb sampling noise without over-allocating arenas.
     */
    static auto compute_max_restrict_level(
        const vertex_num_t total_vertices, const ratio_t sample_ratio
    ) -> layer_num_t {
        if (total_vertices == 0 || sample_ratio <= ratio_t(0) || sample_ratio >= ratio_t(1)) {
            return layer_num_t(1);
        }
        const double total_as_double    = static_cast<double>(total_vertices);
        const double min_cap_as_double  = static_cast<double>(IndexTraitsT::min_layer_cap);
        if (total_as_double <= min_cap_as_double) return layer_num_t(1);
        const double inverse_sample_ratio   = 1.0 / static_cast<double>(sample_ratio);
        // levels_until_min_cap = log_{1/sample_ratio}(total_vertices / min_layer_cap)
        // — the number of pull-out rounds before the expected layer
        // size dips below min_layer_cap.
        const double levels_until_min_cap = std::log(total_as_double / min_cap_as_double) /
                                            std::log(inverse_sample_ratio);
        // +1 of slack so a vertex that happens to land in the very top
        // bucket still has somewhere to go if min_layer_cap drops
        // slightly between runs.
        const layer_num_t max_restrict_level =
            static_cast<layer_num_t>(std::ceil(levels_until_min_cap)) + 1;
        return std::max<layer_num_t>(max_restrict_level, layer_num_t(1));
    }

    /**
     * @brief Construct an empty hier_conv_graph index.
     *
     * @param total_vertices    Expected eventual base-set size. Drives
     *                          max_restrict_level and the per-arena
     *                          slot capacity inside HierarchicalGraph.
     * @param hierarchy_config  ul / bl capacities + sample_ratio.
     * @param propagate_config  conv_graph propagate config consumed by refine_layer.
     * @param pruning_config    conv_graph pruning config consumed by refine_layer.
     */
    IndexStructure(
        const vertex_num_t       total_vertices,
        const hierarchy_config_t hierarchy_config,
        const propagate_config_t propagate_config,
        const pruning_config_t   pruning_config
    ) :
        _max_restrict_level(
            compute_max_restrict_level(total_vertices, hierarchy_config.sample_ratio())),
        _hierarchical_graph(std::make_unique<hierarchical_graph_t>(
            _max_restrict_level,
            hierarchy_config.ul_max_nbr_size(),
            hierarchy_config.bl_max_nbr_size(),
            total_vertices)),
        _hierarchy_config(hierarchy_config),
        _propagate_config(propagate_config),
        _pruning_config(pruning_config),
        _ul_refining_layer_config(
            refining_max_nbr_size_for(hierarchy_config.ul_max_nbr_size())),
        _bl_refining_layer_config(
            refining_max_nbr_size_for(hierarchy_config.bl_max_nbr_size())),
        _vecs_storage()
    {}

    IndexStructure(const IndexStructure&)            = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&)                 = delete;
    IndexStructure& operator=(IndexStructure&&)      = delete;

    // ================================================================
    //   Composed-graph accessor
    // ================================================================

    __attribute__((always_inline))
    auto get_hierarchical_graph() -> hierarchical_graph_t& { return *_hierarchical_graph; }

    __attribute__((always_inline))
    auto get_hierarchical_graph() const -> const hierarchical_graph_t& { return *_hierarchical_graph; }

    // ================================================================
    //   Hierarchical-graph API forwarders
    // ================================================================

    __attribute__((always_inline))
    auto add_vertices(const vertex_num_t n) -> vertex_id_t { return _hierarchical_graph->add_vertices(n); }

    __attribute__((always_inline))
    auto assign_layer(const vertex_id_t vid, const layer_id_t h) -> void {
        _hierarchical_graph->assign_layer(vid, h);
    }

    __attribute__((always_inline))
    auto fetch_layer_nbrs(const vertex_id_t vid, const layer_id_t l) -> std::span<nbr_t> {
        return _hierarchical_graph->fetch_layer_nbrs(vid, l);
    }

    __attribute__((always_inline))
    auto fetch_layer_nbrs(const vertex_id_t vid, const layer_id_t l) const -> std::span<const nbr_t> {
        return _hierarchical_graph->fetch_layer_nbrs(vid, l);
    }

    template <typename FnT>
    __attribute__((always_inline))
    auto with_locked_nbrs(const vertex_id_t vid, const layer_id_t l, FnT&& fn) -> void {
        _hierarchical_graph->with_locked_nbrs(vid, l, std::forward<FnT>(fn));
    }

    __attribute__((always_inline))
    auto num_valid_nbrs(const vertex_id_t vid, const layer_id_t l) const -> vertex_num_t {
        return _hierarchical_graph->num_valid_nbrs(vid, l);
    }

    __attribute__((always_inline))
    auto get_highest_level_id(const vertex_id_t vid) const -> layer_id_t {
        return _hierarchical_graph->get_highest_level_id(vid);
    }

    __attribute__((always_inline))
    auto get_vids_with_highest_level(const layer_id_t h) const
        -> const tbb::concurrent_vector<vertex_id_t>&
    {
        return _hierarchical_graph->get_vids_with_highest_level(h);
    }

    __attribute__((always_inline))
    auto top_occupied_level_id() const -> layer_id_t { return _hierarchical_graph->top_occupied_level_id(); }

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t { return _hierarchical_graph->get_num_vertices(); }

    __attribute__((always_inline))
    auto ul_max_nbr_size() const -> vertex_num_t { return _hierarchical_graph->ul_max_nbr_size(); }

    __attribute__((always_inline))
    auto bl_max_nbr_size() const -> vertex_num_t { return _hierarchical_graph->bl_max_nbr_size(); }

    __attribute__((always_inline))
    auto max_nbr_size(const layer_id_t l) const -> vertex_num_t { return _hierarchical_graph->max_nbr_size(l); }

    __attribute__((always_inline))
    auto max_restrict_level() const -> layer_num_t { return _max_restrict_level; }

    // ================================================================
    //   Refinement-time configs
    // ================================================================

    __attribute__((always_inline))
    auto hierarchy_config() const -> const hierarchy_config_t& { return _hierarchy_config; }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& { return _pruning_config; }

    /** @brief Layer config for refining upper layers (level_id > 0). */
    __attribute__((always_inline))
    auto ul_refining_layer_config() const -> const layer_config_t& { return _ul_refining_layer_config; }

    __attribute__((always_inline))
    auto ul_refining_layer_config() -> layer_config_t& { return _ul_refining_layer_config; }

    /** @brief Layer config for refining the bottom layer (L0). */
    __attribute__((always_inline))
    auto bl_refining_layer_config() const -> const layer_config_t& { return _bl_refining_layer_config; }

    __attribute__((always_inline))
    auto bl_refining_layer_config() -> layer_config_t& { return _bl_refining_layer_config; }

    /** @brief Refining layer config for @p level_id (bl for L0, ul elsewhere). */
    __attribute__((always_inline))
    auto refining_layer_config(const layer_id_t level_id) -> layer_config_t& {
        return (level_id == 0) ? _bl_refining_layer_config : _ul_refining_layer_config;
    }

    // ================================================================
    //   Owned vector storage
    // ================================================================

    __attribute__((always_inline))
    auto get_vecs_storage() -> vecs_storage_t& { return _vecs_storage; }

    __attribute__((always_inline))
    auto get_vecs_storage() const -> const vecs_storage_t& { return _vecs_storage; }

    /**
     * @brief Append a batch of vectors to the owned vector storage.
     *        Delegates to VectorArray::append_batch.
     */
    __attribute__((always_inline))
    auto append_vecs(vector_array_t&& batch_vecs) -> void {
        _vecs_storage.append_batch(std::move(batch_vecs));
    }

    /**
     * @brief Free the composed dynamic::HierarchicalGraph. After this
     *        call subsequent graph-side calls dereference a null
     *        pointer. Configs and _vecs_storage are retained.
     */
    __attribute__((always_inline))
    auto release() -> void { _hierarchical_graph.reset(); }

private:
    /// @brief Inclusive upper bound on highest_level_id. Stored as a
    ///        field because the HierarchicalGraph ctor needs it.
    layer_num_t _max_restrict_level;

    /// @brief Composed HierarchicalGraph (unique_ptr: non-movable).
    std::unique_ptr<hierarchical_graph_t> _hierarchical_graph;

    /// @brief ul / bl capacities + sample_ratio.
    hierarchy_config_t _hierarchy_config;

    /// @brief conv_graph propagate config (per-layer refinement).
    propagate_config_t _propagate_config;

    /// @brief conv_graph pruning config (per-layer refinement).
    pruning_config_t _pruning_config;

    /// @brief RefiningGraph sizing config for level_id > 0.
    layer_config_t _ul_refining_layer_config;

    /// @brief RefiningGraph sizing config for level_id == 0.
    layer_config_t _bl_refining_layer_config;

    /// @brief Owned vector storage. Grown in-place by append_vecs.
    vecs_storage_t _vecs_storage;

};  // class IndexStructure

}   // namespace hier_conv_graph
}   // namespace cpu
}   // namespace artea
