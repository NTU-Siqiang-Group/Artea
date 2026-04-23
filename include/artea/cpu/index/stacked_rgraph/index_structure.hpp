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
 * @FilePath: /Artea/include/artea/cpu/index/stacked_rgraph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Data structure of the dynamic Stacked R-Net index.
 *               Composes a dynamic::HierarchicalGraph with the r-net
 *               configuration and the owned base-vector storage.
 *               Forwards the hierarchical-graph API (add_vertices /
 *               assign_layer / fetch_layer_nbrs / with_locked_nbrs /
 *               per-level bucket and slot queries) so IndexFactory can
 *               operate exclusively against this class.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <tbb/concurrent_vector.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Dynamic hierarchical r-net index.
 *
 * Owns:
 *   - A composed @c dynamic::HierarchicalGraph (held via
 *     @c std::unique_ptr because the graph is non-movable).
 *   - The r-net @c RGraphConfig (beta / L0_radius / max_nbr_size / ...).
 *   - An owned copy of every base vector inserted via
 *     @c append_vecs (used by @c IndexFactory to compute distances
 *     without holding the caller's input batch).
 *
 * Copy / move are both deleted — clients keep an instance inside a
 * @c std::unique_ptr.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure {

public:
    // Public so the dynamic_layer_range adapter (and any other code that
    // treats IndexStructure as a hierarchical-graph forwarder) can read
    // these typedefs without being a friend.
    using vertex_num_t         = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t          = typename IndexTraitsT::vertex_id_t;
    using layer_num_t          = typename IndexTraitsT::layer_num_t;
    using layer_id_t           = typename IndexTraitsT::layer_id_t;
    using distance_t           = typename IndexTraitsT::distance_t;
    using ratio_t              = typename IndexTraitsT::ratio_t;
    using nbr_t                = typename IndexTraitsT::nbr_t;
    using vector_array_t       = typename IndexTraitsT::vector_array_t;
    using vecs_storage_t       = typename IndexTraitsT::vecs_storage_t;
    using rgraph_config_t      = typename IndexTraitsT::stacked_rgraph::rgraph_config_t;
    using pruning_config_t     = typename IndexTraitsT::stacked_rgraph::pruning_config_t;

    using hierarchical_graph_t = typename IndexTraitsT::dynamic::hierarchical_graph_t;

    /** @brief Forwards the dynamic-mode flag from the composed graph so
     *         router-side @c detail::make_layer_range can pick the right
     *         NeighborRange adapter when handed an IndexStructure
     *         directly (as @c stacked_rgraph::IndexFactory does on the
     *         insertion hot path). */
    static constexpr bool is_compacted = hierarchical_graph_t::is_compacted;

    static constexpr layer_id_t unassigned_highest_level_id =
        hierarchical_graph_t::unassigned_highest_level_id;

    /**
     * @brief Construct an empty IndexStructure.
     *
     * @param total_vertices  Expected eventual size of the base dataset.
     *                        Used to derive the hierarchy's hard layer
     *                        cap via @c rgraph_config_t::compute_max_restrict_level.
     * @param rgraph_config   R-graph configuration (r-net geometry,
     *                        queue sizes, neighbor capacity).
     * @param pruning_config  Insert-time RNG pruning policy applied to
     *                        upper-layer (L1+) edges. L0 retains plain
     *                        RNG regardless of this value. Only
     *                        @c scale_coeffs is consumed (default 1.1);
     *                        @c shifted_coeffs is ignored on the
     *                        insertion path.
     */
    IndexStructure(
        const vertex_num_t      total_vertices,
        const rgraph_config_t&  rgraph_config,
        const pruning_config_t  pruning_config = pruning_config_t(
            ratio_t(1.1), ratio_t(0))
    ) :
        _max_restrict_level(
            rgraph_config_t::compute_max_restrict_level(total_vertices)),
        _hierarchical_graph(std::make_unique<hierarchical_graph_t>(
            // max_restrict_level == paper's max_restrict_level.
            // Valid highest_level_id values: [0, max_restrict_level].
            //   - 0                 = vertex participates only at the
            //                         base (level 0).
            //   - 1..max_restrict   = also participates in upper r-net
            //                         levels 1..max_restrict.
            // Upper layers use ul_max_nbr_size; L0 uses bl_max_nbr_size —
            // fully independent, no hardcoded ratio.
            _max_restrict_level,
            rgraph_config.ul_max_nbr_size(),
            rgraph_config.bl_max_nbr_size(),
            total_vertices)),
        _rgraph_config(rgraph_config),
        _pruning_config(pruning_config),
        _vecs_storage()
    {}

    IndexStructure(const IndexStructure&)            = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&)                 = delete;
    IndexStructure& operator=(IndexStructure&&)      = delete;

    // =================================================================
    //   Composed-graph accessor
    // =================================================================

    __attribute__((always_inline))
    auto get_hierarchical_graph() -> hierarchical_graph_t& {
        return *_hierarchical_graph;
    }

    __attribute__((always_inline))
    auto get_hierarchical_graph() const -> const hierarchical_graph_t& {
        return *_hierarchical_graph;
    }

    // =================================================================
    //   Hierarchical-graph API forwarders
    // =================================================================

    __attribute__((always_inline))
    auto add_vertices(const vertex_num_t n) -> vertex_id_t {
        return _hierarchical_graph->add_vertices(n);
    }

    __attribute__((always_inline))
    auto assign_layer(const vertex_id_t vid, const layer_id_t h) -> void {
        _hierarchical_graph->assign_layer(vid, h);
    }

    __attribute__((always_inline))
    auto fetch_layer_nbrs(const vertex_id_t vid, const layer_id_t l)
        -> std::span<nbr_t>
    {
        return _hierarchical_graph->fetch_layer_nbrs(vid, l);
    }

    __attribute__((always_inline))
    auto fetch_layer_nbrs(const vertex_id_t vid, const layer_id_t l) const
        -> std::span<const nbr_t>
    {
        return _hierarchical_graph->fetch_layer_nbrs(vid, l);
    }

    template <typename FnT>
    __attribute__((always_inline))
    auto with_locked_nbrs(const vertex_id_t vid, const layer_id_t l, FnT&& fn)
        -> void
    {
        _hierarchical_graph->with_locked_nbrs(
            vid, l, std::forward<FnT>(fn));
    }

    __attribute__((always_inline))
    auto num_valid_nbrs(const vertex_id_t vid, const layer_id_t l) const
        -> vertex_num_t
    {
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
    auto top_occupied_level_id() const -> layer_id_t {
        return _hierarchical_graph->top_occupied_level_id();
    }

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _hierarchical_graph->get_num_vertices();
    }

    /** @brief Per-vertex neighbor capacity at every upper layer. */
    __attribute__((always_inline))
    auto ul_max_nbr_size() const -> vertex_num_t {
        return _hierarchical_graph->ul_max_nbr_size();
    }

    /** @brief Per-vertex neighbor capacity at the bottom layer (L0). */
    __attribute__((always_inline))
    auto bl_max_nbr_size() const -> vertex_num_t {
        return _hierarchical_graph->bl_max_nbr_size();
    }

    /** @brief Per-vertex capacity at @p l (bl for L0, ul elsewhere). */
    __attribute__((always_inline))
    auto max_nbr_size(const layer_id_t l) const -> vertex_num_t {
        return _hierarchical_graph->max_nbr_size(l);
    }

    // =================================================================
    //   Config accessors
    // =================================================================

    __attribute__((always_inline)) auto rgraph_config()      const -> const rgraph_config_t&  { return _rgraph_config; }
    __attribute__((always_inline)) auto pruning_config()     const -> const pruning_config_t& { return _pruning_config; }
    __attribute__((always_inline)) auto rnet_beta()          const -> ratio_t      { return _rgraph_config.rnet_beta(); }
    __attribute__((always_inline)) auto L0_rnet_radius()     const -> distance_t   { return _rgraph_config.L0_rnet_radius(); }
    __attribute__((always_inline)) auto max_restrict_level() const -> layer_num_t  { return _max_restrict_level; }
    __attribute__((always_inline)) auto search_nn_qs()       const -> vertex_num_t { return _rgraph_config.search_nn_qs(); }
    __attribute__((always_inline)) auto ul_select_nbrs_qs()  const -> vertex_num_t { return _rgraph_config.ul_select_nbrs_qs(); }
    __attribute__((always_inline)) auto bl_select_nbrs_qs()  const -> vertex_num_t { return _rgraph_config.bl_select_nbrs_qs(); }
    static constexpr ratio_t      layer_cap_decay_ratio = rgraph_config_t::layer_cap_decay_ratio;

    /**
     * @brief Covering radius for 0-indexed layer @p h: R_h = L0 * beta^h.
     *        @p h must be in @c [0, max_restrict_level].
     */
    __attribute__((always_inline))
    auto radius_at(const layer_id_t h) const -> distance_t {
        return _rgraph_config.radius_at(h);
    }

    // =================================================================
    //   Owned vector storage
    // =================================================================

    __attribute__((always_inline)) auto get_vecs_storage()       -> vecs_storage_t&       { return _vecs_storage; }
    __attribute__((always_inline)) auto get_vecs_storage() const -> const vecs_storage_t& { return _vecs_storage; }

    /**
     * @brief Append a batch of vectors to the owned vector storage.
     *
     * Delegates to @c VectorArray::append_batch: if the owned storage
     * is currently empty, @p batch_vecs is moved in wholesale; otherwise
     * its contents are copied to the tail in parallel.
     */
    __attribute__((always_inline))
    auto append_vecs(vector_array_t&& batch_vecs) -> void {
        _vecs_storage.append_batch(std::move(batch_vecs));
    }

private:
    /// @brief Hard cap on the number of upper layers (paper 1-indexed).
    ///        Stored as a field because we need it before the
    ///        hierarchical_graph constructor runs.
    layer_num_t _max_restrict_level;

    /// @brief Composed HierarchicalGraph (unique_ptr: non-movable).
    std::unique_ptr<hierarchical_graph_t> _hierarchical_graph;

    /// @brief R-graph configuration.
    rgraph_config_t _rgraph_config;

    /// @brief Insert-time upper-layer pruning configuration.
    pruning_config_t _pruning_config;

    /// @brief Owned vector storage. Grown in-place by append_vecs.
    vecs_storage_t _vecs_storage;

};  // class IndexStructure

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
