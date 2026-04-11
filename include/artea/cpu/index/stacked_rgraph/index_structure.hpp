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
 * @Description: Data structure of the dynamic Stacked R-Net index. Holds
 *               configuration, the layered hierarchy state, and the owned
 *               base-vector storage. The underlying @c HierarchicalGraph
 *               is held by composition (via @c std::unique_ptr, because
 *               @c HierarchicalGraph is move-deleted), and its public API
 *               is forwarded so existing call sites keep working.
 *               Construction and dynamic insertion are performed by
 *               @c stacked_rgraph::IndexFactory.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Dynamic hierarchical r-net index: composes a @c HierarchicalGraph
 *        (the layer storage) with the per-graph configuration and the
 *        owned vector storage used by the insertion algorithm.
 *        Construction and dynamic insertion are performed by
 *        @c stacked_rgraph::IndexFactory.
 *
 * This class used to inherit from @c HierarchicalGraph. It now holds it
 * through a @c std::unique_ptr (HierarchicalGraph is copy-/move-deleted
 * because it contains a @c std::atomic and a @c std::mutex). Exposing
 * the composed graph via @c get_hierarchical_graph() makes it cheap to
 * hand the raw graph to another subsystem without dragging the r-net
 * configuration along. The common @c HierarchicalGraph public API is
 * also forwarded below so direct call sites like
 * @c index.get_num_layers() keep compiling unchanged.
 *
 * Each upper-layer vertex stores its dual identity via @c lnbr_t
 * (@c base_vid is the position in the owned @c _vecs_storage; @c layer_vid
 * is the position in the layer's @c InternalGraph).
 *
 * Per-layer covering radius: @c R_h = L1_rnet_radius * rnet_beta^(h-1),
 * with @c h = 1 being the lowest upper layer (layer_id == 0 internally).
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure {

    using vertex_num_t         = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t          = typename IndexTraitsT::vertex_id_t;
    using layer_num_t          = typename IndexTraitsT::layer_num_t;
    using layer_id_t           = typename IndexTraitsT::layer_id_t;
    using distance_t           = typename IndexTraitsT::distance_t;
    using ratio_t              = typename IndexTraitsT::ratio_t;
    using internal_graph_t     = typename IndexTraitsT::dynamic::internal_graph_t;
    using vector_array_t       = typename IndexTraitsT::vector_array_t;
    using vecs_storage_t       = typename IndexTraitsT::vecs_storage_t;
    using rgraph_config_t      = typename IndexTraitsT::stacked_rgraph::rgraph_config_t;

    using hierarchical_graph_t = typename IndexTraitsT::dynamic::hierarchical_graph_t;

public:
    /**
     * @brief Construct an empty IndexStructure.
     *
     * @param total_vertices  Expected size of the caller's base dataset.
     *                        Used to derive the hierarchy's hard layer cap.
     * @param config          R-graph configuration (r-net geometry, queue
     *                        sizes, neighbor capacity, layer policy).
     */
    IndexStructure(
        const vertex_num_t total_vertices,
        const rgraph_config_t& config
    ) :
        _hierarchical_graph(std::make_unique<hierarchical_graph_t>(
            rgraph_config_t::compute_max_restrict_level(total_vertices))),
        _config(config),
        _max_restrict_level(rgraph_config_t::compute_max_restrict_level(total_vertices)),
        _vecs_storage()
    {
        // Pre-allocate all layers up to max_restrict_level (initially empty).
        // Each layer's InternalGraph starts with capacity derived from
        // the decay ratio; it will auto-resize (2x) if exceeded.
        _hierarchical_graph->grow_layers(
            _max_restrict_level,
            [&](const layer_id_t layer_id) {
                const vertex_num_t cap =
                    _config.capacity_for_layer(layer_id, total_vertices);
                return std::make_unique<internal_graph_t>(
                    cap, _config.max_nbr_size());
            });
    }

    // Copy and move both deleted: the composed HierarchicalGraph is
    // non-movable (atomic + mutex members), so moving IndexStructure
    // would leave the graph behind. Clients should keep IndexStructure
    // inside a std::unique_ptr, as the test and factory already do.
    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&) = delete;
    IndexStructure& operator=(IndexStructure&&) = delete;

    // --- Composed graph accessor ---

    /**
     * @brief Access the underlying @c HierarchicalGraph instance. Use
     *        this to hand the raw graph to another subsystem (e.g.
     *        a serializer or a query-path router) without dragging the
     *        r-net configuration knobs along.
     */
    __attribute__((always_inline))
    auto get_hierarchical_graph() -> hierarchical_graph_t& {
        return *_hierarchical_graph;
    }

    __attribute__((always_inline))
    auto get_hierarchical_graph() const -> const hierarchical_graph_t& {
        return *_hierarchical_graph;
    }

    // --- HierarchicalGraph public API forwarders ---
    //
    // These keep call sites that previously relied on inheritance working
    // unchanged. Every forwarder is a thin inline call to the matching
    // method on @c *_hierarchical_graph.

    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t {
        return _hierarchical_graph->get_num_layers();
    }

    __attribute__((always_inline))
    auto max_layers() const -> layer_num_t {
        return _hierarchical_graph->max_layers();
    }

    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) -> internal_graph_t& {
        return _hierarchical_graph->get_layer_graph(layer_id);
    }

    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) const -> const internal_graph_t& {
        return _hierarchical_graph->get_layer_graph(layer_id);
    }

    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id,
                         std::unique_ptr<internal_graph_t> graph) -> void {
        _hierarchical_graph->set_layer_graph(layer_id, std::move(graph));
    }

    __attribute__((always_inline))
    auto commit_layer(const layer_id_t layer_id) -> void {
        _hierarchical_graph->commit_layer(layer_id);
    }

    template <typename LayerFactoryFnT>
    auto grow_layers(const layer_num_t target_num_layers,
                     LayerFactoryFnT&& layer_factory) -> layer_num_t {
        return _hierarchical_graph->grow_layers(
            target_num_layers,
            std::forward<LayerFactoryFnT>(layer_factory));
    }

    auto trim_empty_layers() -> void {
        _hierarchical_graph->trim_empty_layers();
    }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<internal_graph_t>>& {
        return _hierarchical_graph->get_layer_graphs();
    }

    __attribute__((always_inline))
    auto get_layer_graphs() const
        -> const std::vector<std::unique_ptr<internal_graph_t>>& {
        return _hierarchical_graph->get_layer_graphs();
    }

    // --- Config accessors (delegated to rgraph_config_t) ---

    __attribute__((always_inline)) auto config()             const -> const rgraph_config_t& { return _config; }
    __attribute__((always_inline)) auto rnet_beta()          const -> ratio_t      { return _config.rnet_beta(); }
    __attribute__((always_inline)) auto L1_rnet_radius()     const -> distance_t   { return _config.L1_rnet_radius(); }
    __attribute__((always_inline)) auto max_nbr_size()       const -> vertex_num_t { return _config.max_nbr_size(); }
    __attribute__((always_inline)) auto max_restrict_level() const -> layer_num_t  { return _max_restrict_level; }
    __attribute__((always_inline)) auto search_nn_qs()       const -> vertex_num_t { return _config.search_nn_qs(); }
    __attribute__((always_inline)) auto select_nbrs_qs()     const -> vertex_num_t { return _config.select_nbrs_qs(); }
    __attribute__((always_inline)) auto layer_cap_decay_ratio()    const -> ratio_t      { return _config.layer_cap_decay_ratio(); }
    __attribute__((always_inline)) auto min_layer_cap()      const -> vertex_num_t { return _config.min_layer_cap(); }

    // --- Owned vector storage ---

    /**
     * @brief Access the owned vector storage. The index stores the full
     *        base dataset internally so that incremental @c add_vertices
     *        calls can append new batches in-place; all @c base_vid values
     *        referenced by the hierarchy index into this array.
     */
    __attribute__((always_inline)) auto get_vecs_storage()       -> vecs_storage_t&       { return _vecs_storage; }
    __attribute__((always_inline)) auto get_vecs_storage() const -> const vecs_storage_t& { return _vecs_storage; }

    /**
     * @brief Append a batch of vectors to the owned vector storage.
     *
     * Delegates to @c VectorArray::append_batch: if the owned storage is
     * currently empty, @p batch_vecs is moved in wholesale; otherwise the
     * new vectors are copied in parallel to the end of the existing
     * storage. After this call the new vertices occupy base_vid range
     * @c [old_size, old_size + batch_size).
     */
    __attribute__((always_inline))
    auto append_vecs(vector_array_t&& batch_vecs) -> void {
        _vecs_storage.append_batch(std::move(batch_vecs));
    }

    /**
     * @brief Covering radius for 1-indexed layer @p h (paper convention).
     *        @p h must be in [1, max_restrict_level].
     */
    __attribute__((always_inline))
    auto radius_at(const layer_id_t h) const -> distance_t {
        return _config.radius_at(h);
    }

    __attribute__((always_inline))
    auto capacity_for_layer(const layer_id_t layer_id,
                            const vertex_num_t base_capacity) const -> vertex_num_t {
        return _config.capacity_for_layer(layer_id, base_capacity);
    }

private:
    /// @brief Composed HierarchicalGraph (unique_ptr: non-movable).
    std::unique_ptr<hierarchical_graph_t> _hierarchical_graph;

    /// @brief R-graph configuration (r-net geometry, queue sizes, etc.).
    rgraph_config_t _config;

    /// @brief Hard cap on the number of upper layers, derived from total_vertices.
    layer_num_t _max_restrict_level;

    /// @brief Owned vector storage. Grown in-place by append_vecs.
    vecs_storage_t _vecs_storage;
};

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
