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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <artea/common/logger.hpp>
#include <artea/cpu/index/hierarchical_graph.hpp>

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
    using internal_graph_t     = typename IndexTraitsT::internal_graph_t;
    using vector_array_t       = typename IndexTraitsT::vector_array_t;
    using vecs_storage_t       = typename IndexTraitsT::vecs_storage_t;

    using hierarchical_graph_t = HierarchicalGraph<IndexTraitsT>;

public:
    /**
     * @brief Construct an empty IndexStructure.
     *
     * The composed @c HierarchicalGraph is created with capacity
     * @c _compute_max_restrict_level(total_vertices); the owned vector
     * storage @c _vecs_storage starts empty and is grown in-place by
     * @c append_vecs on every incremental build pass.
     *
     * @param total_vertices       Expected size of the caller's base dataset
     *                             (i.e. layer 0). Used to derive the
     *                             hierarchy's hard layer cap.
     * @param rnet_beta            Radius growth factor: R_h = L1_rnet_radius * rnet_beta^(h-1).
     * @param L1_rnet_radius       Covering radius for layer 1 (the lowest upper layer).
     * @param search_nn_qs         Beam-search queue size used during Phase 1
     *                             top-down nearest-neighbor descent. Also
     *                             the number of uniformly-random vertices
     *                             drawn from the top layer to seed the
     *                             first beam search (no entry point is
     *                             maintained).
     * @param select_nbrs_qs       Beam-search queue size used during Phase 2
     *                             candidate gathering before neighbor pruning.
     * @param max_nbr_size         Per-vertex neighbor capacity for every layer.
     * @param layer_cap_ratio      Geometric decay base for per-layer capacity.
     *                             A value of @c 1 (the default) means
     *                             "no decay" — every layer is sized to N.
     * @param min_layer_cap        Floor on per-layer capacity.
     */
    IndexStructure(
        const vertex_num_t total_vertices,
        const ratio_t rnet_beta,
        const distance_t L1_rnet_radius,
        const vertex_num_t search_nn_qs,
        const vertex_num_t select_nbrs_qs,
        const vertex_num_t max_nbr_size = 32,
        const ratio_t layer_cap_ratio = ratio_t(1),
        const vertex_num_t min_layer_cap = 1024
    ) :
        _hierarchical_graph(std::make_unique<hierarchical_graph_t>(
            _compute_max_restrict_level(total_vertices))),
        _rnet_beta(rnet_beta),
        _L1_rnet_radius(L1_rnet_radius),
        _max_restrict_level(_compute_max_restrict_level(total_vertices)),
        _search_nn_qs(search_nn_qs),
        _select_nbrs_qs(select_nbrs_qs),
        _max_nbr_size(max_nbr_size),
        _layer_cap_ratio(layer_cap_ratio),
        _min_layer_cap(min_layer_cap),
        _vecs_storage()
    {
        if (rnet_beta <= ratio_t(1)) {
            ARTEA_ERROR(fmt::format("rnet_beta ({}) must be > 1", rnet_beta));
        }
        if (L1_rnet_radius <= distance_t(0)) {
            ARTEA_ERROR(fmt::format("L1_rnet_radius ({}) must be > 0", L1_rnet_radius));
        }
        if (search_nn_qs < 1) {
            ARTEA_ERROR(fmt::format("search_nn_qs ({}) must be >= 1", search_nn_qs));
        }
        if (select_nbrs_qs < 1) {
            ARTEA_ERROR(fmt::format("select_nbrs_qs ({}) must be >= 1", select_nbrs_qs));
        }
        if (layer_cap_ratio < ratio_t(1)) {
            ARTEA_ERROR(fmt::format("layer_cap_ratio ({}) must be >= 1", layer_cap_ratio));
        }
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

    template <typename LayerFactoryFnT, typename SeedFnT>
    auto extend_and_seed(const layer_num_t target_num_layers,
                         LayerFactoryFnT&& layer_factory,
                         SeedFnT&& seed_fn) -> bool {
        return _hierarchical_graph->extend_and_seed(
            target_num_layers,
            std::forward<LayerFactoryFnT>(layer_factory),
            std::forward<SeedFnT>(seed_fn));
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

    // --- Config accessors ---

    auto rnet_beta()            const -> ratio_t      { return _rnet_beta; }
    auto L1_rnet_radius()       const -> distance_t   { return _L1_rnet_radius; }
    auto max_nbr_size()         const -> vertex_num_t { return _max_nbr_size; }
    auto max_restrict_level()   const -> layer_num_t  { return _max_restrict_level; }
    auto search_nn_qs()         const -> vertex_num_t { return _search_nn_qs; }
    auto select_nbrs_qs()       const -> vertex_num_t { return _select_nbrs_qs; }
    auto layer_cap_ratio()      const -> ratio_t      { return _layer_cap_ratio; }
    auto min_layer_cap()        const -> vertex_num_t { return _min_layer_cap; }

    // --- Owned vector storage ---

    /**
     * @brief Access the owned vector storage. The index stores the full
     *        base dataset internally so that incremental @c add_vertices
     *        calls can append new batches in-place; all @c base_vid values
     *        referenced by the hierarchy index into this array.
     */
    auto get_vecs_storage()       -> vecs_storage_t&       { return _vecs_storage; }
    auto get_vecs_storage() const -> const vecs_storage_t& { return _vecs_storage; }

    /**
     * @brief Append a batch of vectors to the owned vector storage.
     *
     * Delegates to @c VectorArray::append_batch: if the owned storage is
     * currently empty, @p batch_vecs is moved in wholesale; otherwise the
     * new vectors are copied in parallel to the end of the existing
     * storage. After this call the new vertices occupy base_vid range
     * @c [old_size, old_size + batch_size).
     */
    auto append_vecs(vector_array_t&& batch_vecs) -> void {
        _vecs_storage.append_batch(std::move(batch_vecs));
    }

    /**
     * @brief Covering radius for 1-indexed layer @p h (paper convention).
     *        @p h must be in [1, max_restrict_level].
     */
    auto radius_at(const layer_id_t h) const -> distance_t {
        return static_cast<distance_t>(
            _L1_rnet_radius * std::pow(static_cast<double>(_rnet_beta),
                                       static_cast<double>(h - 1))
        );
    }

    /**
     * @brief Compute the per-layer CSR capacity (number of vertex slots) for
     *        the given 0-indexed layer id.
     *
     * Returns @c base_n when @c layer_cap_ratio == 1 (no decay — every
     * layer pre-allocates for every base vertex, the safest choice).
     * Otherwise returns @c max(base_n / ratio^layer_id, min_layer_cap).
     */
    auto capacity_for_layer(const layer_id_t layer_id,
                            const vertex_num_t base_n) const -> vertex_num_t {
        if (_layer_cap_ratio == ratio_t(1)) {
            return base_n;
        }
        const double divisor = std::pow(static_cast<double>(_layer_cap_ratio),
                                        static_cast<double>(layer_id));
        const double raw = static_cast<double>(base_n) / divisor;
        const vertex_num_t cap = static_cast<vertex_num_t>(raw);
        return std::max<vertex_num_t>(cap, _min_layer_cap);
    }

private:
    /**
     * @brief Compute the recommended @c max_restrict_level for a dataset of
     *        @p total_vertices points as @c ceil(log_10(total_vertices / 1000)).
     *
     * Clamped to at least 1 so the hierarchy can always host one upper layer
     * even for very small datasets. Called from the constructor initializer
     * list; not exposed — callers should not need to know the layer cap.
     */
    static auto _compute_max_restrict_level(const vertex_num_t total_vertices)
        -> layer_num_t
    {
        if (total_vertices == 0) return 1;
        const double ratio = static_cast<double>(total_vertices) / 1000.0;
        if (ratio <= 1.0) return 1;
        const double raw = std::log(ratio) / std::log(10.0);
        const layer_num_t ceiled =
            static_cast<layer_num_t>(std::ceil(raw));
        return std::max<layer_num_t>(ceiled, layer_num_t(1));
    }

    /// @brief Composed @c HierarchicalGraph. Held by @c unique_ptr because
    ///        @c HierarchicalGraph is move-deleted (contains std::atomic
    ///        and std::mutex). Allocated once in the constructor and then
    ///        grown in-place via @c grow_layers / @c extend_and_seed.
    std::unique_ptr<hierarchical_graph_t> _hierarchical_graph;

    /// @brief Radius growth factor between consecutive upper layers:
    ///        @c R_h = L1_rnet_radius * rnet_beta^(h-1). Must be > 1.
    ratio_t      _rnet_beta;

    /// @brief Covering radius of layer 1 (the lowest upper layer, @c h==1).
    ///        Anchors the geometric progression of per-layer radii.
    distance_t   _L1_rnet_radius;

    /// @brief Hard cap on the number of upper layers in the hierarchy,
    ///        derived from @c total_vertices via @c _compute_max_restrict_level.
    layer_num_t  _max_restrict_level;

    /// @brief Beam-search queue size for Phase 1 top-down nearest-neighbor
    ///        descent. Also the number of uniformly-random seed vertices
    ///        drawn from the top layer to start the search.
    vertex_num_t _search_nn_qs;

    /// @brief Beam-search queue size for Phase 2 candidate gathering
    ///        performed before neighbor pruning.
    vertex_num_t _select_nbrs_qs;

    /// @brief Per-vertex neighbor capacity applied uniformly to every layer.
    vertex_num_t _max_nbr_size;

    /// @brief Geometric decay base for per-layer CSR capacity. A value of
    ///        @c 1 disables decay — every layer is sized to the full base
    ///        dataset. Must be >= 1.
    ratio_t      _layer_cap_ratio;

    /// @brief Floor on per-layer CSR capacity; guards against over-shrinking
    ///        upper layers when @c _layer_cap_ratio > 1.
    vertex_num_t _min_layer_cap;

    /// @brief Owned vector storage. Grown in-place by @c append_vecs;
    ///        every @c base_vid in the hierarchy indexes into this array.
    vecs_storage_t _vecs_storage;
};

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
