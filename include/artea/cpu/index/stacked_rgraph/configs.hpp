/*
 * @FilePath: /Artea/include/artea/cpu/index/stacked_rgraph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for Stacked R-Net (dynamic hierarchical r-net).
 */

#pragma once

#include <cmath>
#include <artea/common/logger.hpp>
#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/** @brief Stacked R-Graph reuses the same PruningConfig as conv_graph. */
template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

/**
 * @brief Configuration for the Stacked R-Net hierarchical index.
 *
 * Encapsulates all construction-time parameters: r-net geometry
 * (beta, L1 radius), beam-search queue sizes, per-vertex neighbor
 * capacity, and per-layer pre-allocation policy.
 *
 * @tparam BaseTraitsT The base traits type.
 */
template <typename BaseTraitsT>
struct RGraphConfig {
    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using layer_num_t  = typename BaseTraitsT::layer_num_t;
    using distance_t   = typename BaseTraitsT::distance_t;
    using ratio_t      = typename BaseTraitsT::ratio_t;

    /**
     * @brief Construct a RGraphConfig.
     * @param rnet_beta        Radius growth factor: R_h = L1_rnet_radius * rnet_beta^(h-1). Must be > 1.
     * @param L1_rnet_radius   Covering radius for layer 1 (the lowest upper layer). Must be > 0.
     * @param search_nn_qs     Beam-search queue size for Phase 1 descent. Must be >= 1.
     * @param select_nbrs_qs   Beam-search queue size for Phase 2 candidate gathering. Must be >= 1.
     * @param max_nbr_size     Per-vertex neighbor capacity for every layer (default: 32).
     * @param layer_cap_decay_ratio  Geometric decay ratio for per-layer initial capacity;
     *                               capacity(layer_id) = base_capacity * decay^layer_id.
     *                               Must be in (0, 1]. Default 0.5 (each layer starts
     *                               at half the capacity of the one below).
     * @param min_layer_cap    Floor on per-layer capacity (default: 1024).
     */
    RGraphConfig(
        ratio_t rnet_beta,
        distance_t L1_rnet_radius,
        vertex_num_t search_nn_qs,
        vertex_num_t select_nbrs_qs,
        vertex_num_t max_nbr_size = 32,
        ratio_t layer_cap_decay_ratio = ratio_t(0.5),
        vertex_num_t min_layer_cap = 1024
    ) :
        _rnet_beta(rnet_beta),
        _L1_rnet_radius(L1_rnet_radius),
        _search_nn_qs(search_nn_qs),
        _select_nbrs_qs(select_nbrs_qs),
        _max_nbr_size(max_nbr_size),
        _layer_cap_decay_ratio(layer_cap_decay_ratio),
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
        if (layer_cap_decay_ratio <= ratio_t(0) || layer_cap_decay_ratio > ratio_t(1)) {
            ARTEA_ERROR(fmt::format("layer_cap_decay_ratio ({}) must be in (0, 1]", layer_cap_decay_ratio));
        }
    }

    // Const getters
    __attribute__((always_inline)) auto rnet_beta()          const -> ratio_t      { return _rnet_beta; }
    __attribute__((always_inline)) auto L1_rnet_radius()     const -> distance_t   { return _L1_rnet_radius; }
    __attribute__((always_inline)) auto search_nn_qs()       const -> vertex_num_t { return _search_nn_qs; }
    __attribute__((always_inline)) auto select_nbrs_qs()     const -> vertex_num_t { return _select_nbrs_qs; }
    __attribute__((always_inline)) auto max_nbr_size()       const -> vertex_num_t { return _max_nbr_size; }
    __attribute__((always_inline)) auto layer_cap_decay_ratio()    const -> ratio_t      { return _layer_cap_decay_ratio; }
    __attribute__((always_inline)) auto min_layer_cap()      const -> vertex_num_t { return _min_layer_cap; }

    /**
     * @brief Covering radius for 1-indexed layer @p h: R_h = L1 * beta^(h-1).
     */
    __attribute__((always_inline))
    auto radius_at(const layer_num_t h) const -> distance_t {
        return static_cast<distance_t>(
            _L1_rnet_radius * std::pow(static_cast<double>(_rnet_beta),
                                       static_cast<double>(h - 1))
        );
    }

    /**
     * @brief Initial CSR capacity for the given 0-indexed layer id.
     *        capacity = base_capacity * decay_ratio^layer_id, floored at min_layer_cap.
     *        Returns base_capacity when decay_ratio == 1 (no decay).
     */
    __attribute__((always_inline))
    auto capacity_for_layer(const vertex_num_t layer_id, const vertex_num_t base_capacity) const -> vertex_num_t {
        if (_layer_cap_decay_ratio == ratio_t(1)) return base_capacity;
        const double factor = std::pow(static_cast<double>(_layer_cap_decay_ratio), static_cast<double>(layer_id));
        const double raw = static_cast<double>(base_capacity) * factor;
        const vertex_num_t cap = static_cast<vertex_num_t>(raw);
        return std::max<vertex_num_t>(cap, _min_layer_cap);
    }

    /**
     * @brief Compute the max restrict level for a dataset of @p total_vertices.
     */
    static auto compute_max_restrict_level(const vertex_num_t total_vertices) -> layer_num_t
    {
        if (total_vertices == 0) return 1;
        const double ratio = static_cast<double>(total_vertices) / 1000.0;
        if (ratio <= 1.0) return 1;
        const double raw = std::log(ratio) / std::log(10.0);
        const layer_num_t ceiled = static_cast<layer_num_t>(std::ceil(raw));
        return std::max<layer_num_t>(ceiled, layer_num_t(1));
    }

private:
    /** @brief Radius growth factor: R_h = L1_rnet_radius * rnet_beta^(h-1). Must be > 1. */
    ratio_t      _rnet_beta;

    /** @brief Covering radius for layer 1 (the lowest upper layer, not bottom layer). Must be > 0. */
    distance_t   _L1_rnet_radius;

    /** @brief Beam-search queue size for Phase 1 top-down descent. */
    vertex_num_t _search_nn_qs;

    /** @brief Beam-search queue size for Phase 2 candidate gathering. */
    vertex_num_t _select_nbrs_qs;

    /** @brief Per-vertex neighbor capacity for every layer. */
    vertex_num_t _max_nbr_size;

    /** @brief Geometric decay ratio for per-layer initial capacity; must be in (0, 1]. */
    ratio_t      _layer_cap_decay_ratio;

    /** @brief Floor on per-layer capacity. */
    vertex_num_t _min_layer_cap;
};

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
