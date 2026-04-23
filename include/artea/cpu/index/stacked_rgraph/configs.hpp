/*
 * @FilePath: /Artea/include/artea/cpu/index/stacked_rgraph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for Stacked R-Net (dynamic hierarchical
 *               r-net). PruningConfig is a using-alias of
 *               conv_graph::PruningConfig — the stacked-rgraph insertion
 *               path only reads scale_coeffs from it (shift is a
 *               post-refinement concern and is ignored here).
 */

#pragma once

#include <cmath>
#include <artea/common/logger.hpp>
#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Configuration for the Stacked R-Net hierarchical index.
 *
 * Encapsulates all construction-time parameters: r-net geometry
 * (beta, L1 radius), beam-search queue sizes (split into an upper-layer
 * variant used for L1+ and a bottom-layer variant used for L0),
 * per-vertex neighbor capacity, and per-layer pre-allocation policy.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
struct RGraphConfig {
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using layer_num_t  = typename IndexTraitsT::layer_num_t;
    using distance_t   = typename IndexTraitsT::distance_t;
    using ratio_t      = typename IndexTraitsT::ratio_t;

    /** @brief Geometric decay ratio for per-layer initial capacity;
     *         capacity(layer_id) = base_capacity * decay^layer_id.
     *         Each layer starts at half the capacity of the one below. */
    static constexpr ratio_t      layer_cap_decay_ratio = ratio_t(0.5);

    /**
     * @brief Construct a RGraphConfig.
     * @param rnet_beta           Radius growth factor: R_h = L0_rnet_radius * rnet_beta^h. Must be > 1.
     * @param L0_rnet_radius      Covering radius for layer 0 (the bottom layer). Must be > 0.
     *                            The layer-1 covering radius is derived as @c rnet_beta * @c L0_rnet_radius.
     * @param search_nn_qs        Beam-search queue size for Phase 1 descent. Must be >= 1.
     * @param ul_select_nbrs_qs   Beam-search queue size for Phase 2 candidate gathering at
     *                            upper levels (L1..highest_insert_level). Must be >= 1. Default 100.
     * @param bl_select_nbrs_qs   Beam-search queue size for Phase 2 candidate gathering at
     *                            the bottom level (L0). Must be >= 1. Default 100.
     * @param max_nbr_size        Per-vertex neighbor capacity for every layer (default: 32).
     */
    RGraphConfig(
        ratio_t rnet_beta,
        distance_t L0_rnet_radius,
        vertex_num_t search_nn_qs,
        vertex_num_t ul_select_nbrs_qs = 100,
        vertex_num_t bl_select_nbrs_qs = 100,
        vertex_num_t max_nbr_size = 32
    ) :
        _rnet_beta(rnet_beta),
        _L0_rnet_radius(L0_rnet_radius),
        _search_nn_qs(search_nn_qs),
        _ul_select_nbrs_qs(ul_select_nbrs_qs),
        _bl_select_nbrs_qs(bl_select_nbrs_qs),
        _max_nbr_size(max_nbr_size)
    {
        if (rnet_beta <= ratio_t(1)) {
            ARTEA_ERROR(fmt::format("rnet_beta ({}) must be > 1", rnet_beta));
        }
        if (L0_rnet_radius <= distance_t(0)) {
            ARTEA_ERROR(fmt::format("L0_rnet_radius ({}) must be > 0", L0_rnet_radius));
        }
        if (search_nn_qs < 1) {
            ARTEA_ERROR(fmt::format("search_nn_qs ({}) must be >= 1", search_nn_qs));
        }
        if (ul_select_nbrs_qs < 1) {
            ARTEA_ERROR(fmt::format("ul_select_nbrs_qs ({}) must be >= 1", ul_select_nbrs_qs));
        }
        if (bl_select_nbrs_qs < 1) {
            ARTEA_ERROR(fmt::format("bl_select_nbrs_qs ({}) must be >= 1", bl_select_nbrs_qs));
        }
    }

    // Const getters
    __attribute__((always_inline)) auto rnet_beta()          const -> ratio_t      { return _rnet_beta; }
    __attribute__((always_inline)) auto L0_rnet_radius()     const -> distance_t   { return _L0_rnet_radius; }
    __attribute__((always_inline)) auto search_nn_qs()       const -> vertex_num_t { return _search_nn_qs; }
    __attribute__((always_inline)) auto ul_select_nbrs_qs()  const -> vertex_num_t { return _ul_select_nbrs_qs; }
    __attribute__((always_inline)) auto bl_select_nbrs_qs()  const -> vertex_num_t { return _bl_select_nbrs_qs; }
    __attribute__((always_inline)) auto max_nbr_size()       const -> vertex_num_t { return _max_nbr_size; }

    /**
     * @brief Covering radius for 0-indexed layer @p h: R_h = L0 * beta^h.
     *        Thus R_0 = L0_rnet_radius and R_1 = beta * L0_rnet_radius.
     */
    __attribute__((always_inline))
    auto radius_at(const layer_num_t h) const -> distance_t {
        return static_cast<distance_t>(
            _L0_rnet_radius * std::pow(static_cast<double>(_rnet_beta),
                                       static_cast<double>(h))
        );
    }

    /**
     * @brief Initial CSR capacity for the given 0-indexed layer id.
     *        capacity = base_capacity * decay_ratio^layer_id, floored at
     *        @c IndexTraitsT::min_layer_cap.
     *        Returns base_capacity when decay_ratio == 1 (no decay).
     */
    __attribute__((always_inline))
    auto capacity_for_layer(const vertex_num_t layer_id, const vertex_num_t base_capacity) const -> vertex_num_t {
        if constexpr (layer_cap_decay_ratio == ratio_t(1)) return base_capacity;
        const double factor = std::pow(static_cast<double>(layer_cap_decay_ratio), static_cast<double>(layer_id));
        const double raw = static_cast<double>(base_capacity) * factor;
        const vertex_num_t cap = static_cast<vertex_num_t>(raw);
        return std::max<vertex_num_t>(cap, IndexTraitsT::min_layer_cap);
    }

    /**
     * @brief Compute the max restrict level for a dataset of @p total_vertices.
     */
    static auto compute_max_restrict_level(const vertex_num_t total_vertices) -> layer_num_t
    {
        if (total_vertices == 0) return 1;
        const double ratio = static_cast<double>(total_vertices) / 1000.0;
        if (ratio <= 1.0) return 1;
        const double raw = std::log(ratio);
        const layer_num_t ceiled = static_cast<layer_num_t>(std::ceil(raw));
        return std::max<layer_num_t>(ceiled, layer_num_t(1));
    }

private:
    /** @brief Radius growth factor: R_h = L0_rnet_radius * rnet_beta^h. Must be > 1. */
    ratio_t      _rnet_beta;

    /** @brief Covering radius for layer 0 (the bottom layer). Must be > 0.
     *         Layer 1's covering radius is derived as @c _rnet_beta * @c _L0_rnet_radius. */
    distance_t   _L0_rnet_radius;

    /** @brief Beam-search queue size for Phase 1 top-down descent. */
    vertex_num_t _search_nn_qs;

    /** @brief Beam-search queue size for Phase 2 candidate gathering at upper levels (L1+). */
    vertex_num_t _ul_select_nbrs_qs;

    /** @brief Beam-search queue size for Phase 2 candidate gathering at the bottom level (L0). */
    vertex_num_t _bl_select_nbrs_qs;

    /** @brief Per-vertex neighbor capacity for every layer. */
    vertex_num_t _max_nbr_size;
};

/** @brief stacked_rgraph reuses conv_graph's PruningConfig — the
 *         insertion path reads scale_coeffs only; shifted_coeffs is a
 *         post-refinement concern and is ignored on the insertion path. */
template <typename IndexTraitsT>
using PruningConfig = conv_graph::PruningConfig<IndexTraitsT>;

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
