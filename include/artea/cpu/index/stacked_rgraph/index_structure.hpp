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
 *               configuration and the common dataset-owning base.
 *               Forwards the hierarchical-graph API (add_vertices /
 *               assign_layer / fetch_level_nbrs / with_locked_nbrs /
 *               per-level bucket and slot queries) so IndexFactory can
 *               operate exclusively against this class.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/concurrent_vector.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/index/dataset_index.hpp>
#include <artea/cpu/index/compactor/hierarchical_graph_compactor.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Dynamic hierarchical r-net index.
 *
 * Owns:
 *   - A composed @c dynamic::HierarchicalGraph (held via
 *     @c std::unique_ptr because the graph is non-movable).
 *   - The r-net @c RGraphConfig (rnet_beta / tau_k / tau / l0_min_distance / max_nbr_size / ...).
 *   - Dataset ownership or an explicit borrowed vector view inherited from @c DatasetIndex.
 *
 * Copy / move are both deleted — clients keep an instance inside a
 * @c std::unique_ptr.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure : public DatasetIndex<IndexTraitsT> {
    using data_base_t = DatasetIndex<IndexTraitsT>;

public:
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using compact_graph_t = typename IndexTraitsT::compact::hierarchical_graph_t;

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

    static constexpr layer_id_t invalid_level_id =
        hierarchical_graph_t::invalid_level_id;

    /**
     * @brief Construct an empty IndexStructure.
     *
     * @param total_vertices  Expected eventual size of the base dataset.
     *                        Used to derive the hierarchy's hard layer
     *                        cap via @c rgraph_config_t::compute_max_allowed_level_id.
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
        _max_allowed_level_id(
            rgraph_config_t::compute_max_allowed_level_id(total_vertices)),
        _hierarchical_graph(std::make_unique<hierarchical_graph_t>(
            // max_allowed_level_id corresponds to the paper's max_restrict_level.
            // Valid highest_level_id values: [0, max_allowed_level_id].
            //   - 0                 = vertex participates only at the
            //                         base (level 0).
            //   - 1..max_allowed_level_id = also participates in upper r-net
            //                              levels 1..max_allowed_level_id.
            // Upper layers use ul_max_nbr_size; L0 uses bl_max_nbr_size —
            // fully independent, no hardcoded ratio.
            _max_allowed_level_id,
            rgraph_config.ul_max_nbr_size(),
            rgraph_config.bl_max_nbr_size(),
            total_vertices)),
        _rgraph_config(rgraph_config),
        _pruning_config(pruning_config)
    {}

    /** @brief Own a loaded dataset; call prepare_build() before constructing the graph. */
    explicit IndexStructure(std::unique_ptr<vector_dataset_t> dataset)
        : data_base_t(std::move(dataset)) {}

    IndexStructure(std::unique_ptr<vector_dataset_t> dataset, const rgraph_config_t& rgraph_config,
                   pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0)))
        : data_base_t(std::move(dataset)) {
        prepare_build(rgraph_config, pruning_config);
    }

    IndexStructure(const IndexStructure&)            = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&)                 = delete;
    IndexStructure& operator=(IndexStructure&&)      = delete;

    /** @brief Build against immutable vectors owned by the caller, without copying them.
     *  The vector array must outlive this index and remain unchanged. Appending is disabled. */
    IndexStructure(const vector_array_t& vectors, const rgraph_config_t& rgraph_config,
                   const pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0)))
        : data_base_t(vectors) {
        prepare_build(rgraph_config, pruning_config);
    }

    IndexStructure(vector_array_t&&, const rgraph_config_t&,
                   pruning_config_t = pruning_config_t(ratio_t(1.1), ratio_t(0))) = delete;
    IndexStructure(const vector_array_t&&, const rgraph_config_t&,
                   pruning_config_t = pruning_config_t(ratio_t(1.1), ratio_t(0))) = delete;

    /** @brief Rebuild graph state over the same vectors; invalidates previous graph references.
     *  Call after queries stop. Dataset references survive both rebuilding and allocation failure. */
    auto prepare_build(const rgraph_config_t& rgraph_config,
                       pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0)))
        -> IndexStructure& {
        _compact_graph.reset();
        _hierarchical_graph.reset();
        const auto count = static_cast<vertex_num_t>(this->get_base_vecs().get_num_vecs());
        _max_allowed_level_id = rgraph_config_t::compute_max_allowed_level_id(count);
        auto graph = std::make_unique<hierarchical_graph_t>(
            _max_allowed_level_id, rgraph_config.ul_max_nbr_size(), rgraph_config.bl_max_nbr_size(), count);
        _rgraph_config = rgraph_config;
        _pruning_config = pruning_config;
        _hierarchical_graph = std::move(graph);
        return *this;
    }

    /** @brief Retain the compact graph and dataset, releasing the building graph only after success. */
    template <typename DistFuncT>
    auto compact(const DistFuncT& distance) -> void {
        using compactor_t = typename IndexTraitsT::hierarchical_graph_compactor_t;
        auto graph = compactor_t::compact_graph(get_hierarchical_graph(), this->get_base_vecs(), distance);
        _compact_graph.emplace(std::move(graph));
        _hierarchical_graph.reset();
    }

    /** @brief Search graph, valid until the next rebuild or index destruction. */
    auto get_compact_graph() const -> const compact_graph_t& {
        if (!_compact_graph) throw std::logic_error("No compact graph; call compact first");
        return *_compact_graph;
    }

    // =================================================================
    //   Composed-graph accessor
    // =================================================================

    __attribute__((always_inline))
    auto get_hierarchical_graph() -> hierarchical_graph_t& {
        if (!_hierarchical_graph) throw std::logic_error("No building graph; call prepare_build first");
        return *_hierarchical_graph;
    }

    __attribute__((always_inline))
    auto get_hierarchical_graph() const -> const hierarchical_graph_t& {
        if (!_hierarchical_graph) throw std::logic_error("No building graph; call prepare_build first");
        return *_hierarchical_graph;
    }

    // =================================================================
    //   Hierarchical-graph API forwarders
    // =================================================================

    __attribute__((always_inline))
    auto add_vertices(const vertex_num_t n) -> vertex_id_t {
        return get_hierarchical_graph().add_vertices(n);
    }

    __attribute__((always_inline))
    auto assign_layer(const vertex_id_t vid, const layer_id_t h) -> void {
        get_hierarchical_graph().assign_layer(vid, h);
    }

    __attribute__((always_inline))
    auto fetch_level_nbrs(const vertex_id_t vid, const layer_id_t l)
        -> std::span<nbr_t>
    {
        return get_hierarchical_graph().fetch_level_nbrs(vid, l);
    }

    __attribute__((always_inline))
    auto fetch_level_nbrs(const vertex_id_t vid, const layer_id_t l) const
        -> std::span<const nbr_t>
    {
        return get_hierarchical_graph().fetch_level_nbrs(vid, l);
    }

    template <typename FnT>
    __attribute__((always_inline))
    auto with_locked_nbrs(const vertex_id_t vid, const layer_id_t l, FnT&& fn)
        -> void
    {
        get_hierarchical_graph().with_locked_nbrs(
            vid, l, std::forward<FnT>(fn));
    }

    __attribute__((always_inline))
    auto num_valid_nbrs(const vertex_id_t vid, const layer_id_t l) const
        -> vertex_num_t
    {
        return get_hierarchical_graph().num_valid_nbrs(vid, l);
    }

    __attribute__((always_inline))
    auto get_highest_level_id(const vertex_id_t vid) const -> layer_id_t {
        return get_hierarchical_graph().get_highest_level_id(vid);
    }

    __attribute__((always_inline))
    auto get_vids_with_highest_level(const layer_id_t h) const
        -> const tbb::concurrent_vector<vertex_id_t>&
    {
        return get_hierarchical_graph().get_vids_with_highest_level(h);
    }

    __attribute__((always_inline))
    auto get_top_level_vids() const {
        return get_hierarchical_graph().get_top_level_vids();
    }

    __attribute__((always_inline))
    auto top_occupied_level_id() const -> layer_id_t {
        return get_hierarchical_graph().top_occupied_level_id();
    }

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return get_hierarchical_graph().get_num_vertices();
    }

    /** @brief Per-vertex neighbor capacity at every upper layer. */
    __attribute__((always_inline))
    auto ul_max_nbr_size() const -> vertex_num_t {
        return get_hierarchical_graph().ul_max_nbr_size();
    }

    /** @brief Per-vertex neighbor capacity at the bottom layer (L0). */
    __attribute__((always_inline))
    auto bl_max_nbr_size() const -> vertex_num_t {
        return get_hierarchical_graph().bl_max_nbr_size();
    }

    /** @brief Per-vertex capacity at @p l (bl for L0, ul elsewhere). */
    __attribute__((always_inline))
    auto max_nbr_size(const layer_id_t l) const -> vertex_num_t {
        return get_hierarchical_graph().max_nbr_size(l);
    }

    // =================================================================
    //   Config accessors
    // =================================================================

    __attribute__((always_inline))
    auto rgraph_config() const -> const rgraph_config_t& {
        return _rgraph_config.value();
    }
    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& {
        return _pruning_config.value();
    }
    __attribute__((always_inline))
    auto rnet_beta() const -> ratio_t {
        return _rgraph_config.value().rnet_beta();
    }
    __attribute__((always_inline))
    auto tau_k() const -> ratio_t {
        return _rgraph_config.value().tau_k();
    }
    __attribute__((always_inline))
    auto tau() const -> ratio_t {
        return _rgraph_config.value().tau();
    }
    __attribute__((always_inline))
    auto l0_min_distance() const -> distance_t {
        return _rgraph_config.value().l0_min_distance();
    }
    __attribute__((always_inline))
    auto max_allowed_level_id() const -> layer_num_t { return _max_allowed_level_id; }
    __attribute__((always_inline))
    auto search_nn_qs() const -> vertex_num_t {
        return _rgraph_config.value().search_nn_qs();
    }
    __attribute__((always_inline))
    auto ul_select_nbrs_qs() const -> vertex_num_t {
        return _rgraph_config.value().ul_select_nbrs_qs();
    }
    __attribute__((always_inline))
    auto bl_select_nbrs_qs() const -> vertex_num_t {
        return _rgraph_config.value().bl_select_nbrs_qs();
    }
    static constexpr ratio_t      layer_cap_decay_ratio = rgraph_config_t::layer_cap_decay_ratio;

    /**
     * @brief Upper-layer radius: R_h = l0_min_distance * (1 + tau_k) * rnet_beta^(h - 1).
     *        At L0, returns the characteristic minimum-distance scale.
     *        @p h must be in @c [0, max_allowed_level_id].
     */
    __attribute__((always_inline))
    auto radius_at(const layer_id_t h) const -> distance_t {
        return _rgraph_config.value().radius_at(h);
    }

    /** @brief Append a batch in incremental mode while the building graph is available. */
    auto append_vecs(vector_array_t&& batch_vecs) -> void {
        if (get_num_vertices() != this->get_base_vecs().get_num_vecs()) {
            throw std::logic_error("Build existing vectors before appending another batch");
        }
        this->append_base_vecs(std::move(batch_vecs));
    }

    /** @brief Release the building graph while preserving the dataset and any compact graph. */
    auto release() -> void { _hierarchical_graph.reset(); }

protected:
    /** @brief Publish a completed building graph with its actual layer bound.
     *  Derived bulk builders may need more layers than the insertion heuristic.
     *  The caller must supply a non-null graph using this index's configuration.
     *  Invalidates graph references while preserving vectors and configurations. */
    auto replace_building_graph(std::unique_ptr<hierarchical_graph_t> graph) -> void {
        if (!graph) throw std::invalid_argument("Building graph must not be null");
        _max_allowed_level_id = graph->max_allowed_level_id();
        _compact_graph.reset();
        _hierarchical_graph = std::move(graph);
    }

private:
    /// @brief Inclusive upper bound on the allowed highest_level_id.
    ///        Stored as a field because we need it before the
    ///        hierarchical_graph constructor runs.
    layer_num_t _max_allowed_level_id = 0;

    /// @brief Composed HierarchicalGraph (unique_ptr: non-movable).
    std::unique_ptr<hierarchical_graph_t> _hierarchical_graph;

    /// @brief R-graph configuration.
    std::optional<rgraph_config_t> _rgraph_config;

    /// @brief Insert-time upper-layer pruning configuration.
    std::optional<pruning_config_t> _pruning_config;

    /** @brief Query graph produced by successful compaction; independent of the building graph. */
    std::optional<compact_graph_t> _compact_graph;

};  // class IndexStructure

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
