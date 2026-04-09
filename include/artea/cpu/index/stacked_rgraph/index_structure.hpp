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
 *               configuration and the layered hierarchy state only; the
 *               online insertion algorithm lives in
 *               @c stacked_rgraph::IndexFactory.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <artea/common/logger.hpp>
#include <artea/cpu/index/hierarchical_graph_v2.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Dynamic hierarchical r-net index: holds the layer storage (via
 *        @c HierarchicalGraphV2) and the per-graph configuration used by
 *        the insertion algorithm. Construction and dynamic insertion are
 *        performed by @c stacked_rgraph::IndexFactory.
 *
 * Each upper-layer vertex stores its dual identity via @c lnbr_t
 * (@c base_vid is the position in the caller's base_vecs; @c layer_vid is
 * the position in the layer's @c InternalGraph).
 *
 * Per-layer covering radius: @c R_h = L1_rnet_radius * rnet_beta^(h-1),
 * with @c h = 1 being the lowest upper layer (layer_id == 0 internally).
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure : public HierarchicalGraphV2<IndexTraitsT> {

    using base_t           = HierarchicalGraphV2<IndexTraitsT>;

    using vertex_num_t     = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t      = typename IndexTraitsT::vertex_id_t;
    using layer_num_t      = typename IndexTraitsT::layer_num_t;
    using layer_id_t       = typename IndexTraitsT::layer_id_t;
    using distance_t       = typename IndexTraitsT::distance_t;
    using ratio_t          = typename IndexTraitsT::ratio_t;

    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct an empty IndexStructure.
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
        base_t(_compute_max_restrict_level(total_vertices)),
        _rnet_beta(rnet_beta),
        _L1_rnet_radius(L1_rnet_radius),
        _max_restrict_level(_compute_max_restrict_level(total_vertices)),
        _search_nn_qs(search_nn_qs),
        _select_nbrs_qs(select_nbrs_qs),
        _max_nbr_size(max_nbr_size),
        _layer_cap_ratio(layer_cap_ratio),
        _min_layer_cap(min_layer_cap)
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

    // Inherits copy/move-deleted from HierarchicalGraphV2.

    // --- Config accessors ---

    auto rnet_beta()            const -> ratio_t      { return _rnet_beta; }
    auto L1_rnet_radius()       const -> distance_t   { return _L1_rnet_radius; }
    auto max_nbr_size()         const -> vertex_num_t { return _max_nbr_size; }
    auto max_restrict_level()   const -> layer_num_t  { return _max_restrict_level; }
    auto search_nn_qs()         const -> vertex_num_t { return _search_nn_qs; }
    auto select_nbrs_qs()       const -> vertex_num_t { return _select_nbrs_qs; }
    auto layer_cap_ratio()      const -> ratio_t      { return _layer_cap_ratio; }
    auto min_layer_cap()        const -> vertex_num_t { return _min_layer_cap; }

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

    /**
     * @brief Walk the inter_layer_link chain from @p (layer_idx, layer_vid)
     *        all the way down to layer 0 and return the corresponding
     *        @c base_vid. Returns @c invalid_vertex_id if any step is OOB.
     */
    auto resolve_base_vid(const layer_id_t layer_idx,
                          const vertex_id_t layer_vid) const -> vertex_id_t {
        vertex_id_t cur = layer_vid;
        for (layer_id_t down = layer_idx; down > 0; --down) {
            const auto& layer = this->get_layer_graph(down);
            if (cur >= layer.get_num_vertices()) return invalid_vertex_id;
            cur = layer.get_inter_layer_link(cur);
        }
        const auto& layer0 = this->get_layer_graph(0);
        if (cur >= layer0.get_num_vertices()) return invalid_vertex_id;
        return layer0.get_inter_layer_link(cur);
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
};

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
