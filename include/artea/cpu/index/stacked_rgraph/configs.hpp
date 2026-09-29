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

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <artea/common/logger.hpp>
#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Configuration for the Stacked R-Net hierarchical index.
 *
 * Encapsulates all construction-time parameters: r-net geometry
 * (beta, tau_k, L0 minimum distance), beam-search queue sizes (split into an upper-layer
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
     * @param rnet_beta           Radius growth factor. Must be finite and > 1.
     * @param tau_k              Nonnegative radius coefficient, independent of the pruning shift tau.
     *                           R_1 = l0_min_distance * (1 + tau_k).
     * @param tau                Shift coefficient from shifted_coeffs, finite and >= 0.
     *                           Retained for compatibility; does not affect r-net radii.
     * @param l0_min_distance     Characteristic L0 minimum distance in build-distance units. Must be > 0.
     * @param search_nn_qs        Beam-search queue size for Phase 1 descent. Must be >= 1.
     * @param ul_select_nbrs_qs   Beam-search queue size for Phase 2 candidate gathering at
     *                            upper levels (L1..highest_insert_level). Must be >= 1. Default 100.
     * @param bl_select_nbrs_qs   Beam-search queue size for Phase 2 candidate gathering at
     *                            the bottom level (L0). Must be >= 1. Default 100.
     * @param ul_max_nbr_size     Per-vertex neighbor capacity at every upper
     *                            layer (level_id > 0). Default 32.
     * @param bl_max_nbr_size     Per-vertex neighbor capacity at the bottom
     *                            layer (L0). Independent of the upper value.
     *                            Default 64.
     */
    RGraphConfig(
        ratio_t rnet_beta,
        ratio_t tau_k,
        ratio_t tau,
        distance_t l0_min_distance,
        vertex_num_t search_nn_qs,
        vertex_num_t ul_select_nbrs_qs = 100,
        vertex_num_t bl_select_nbrs_qs = 100,
        vertex_num_t ul_max_nbr_size = 32,
        vertex_num_t bl_max_nbr_size = 64
    ) :
        _rnet_beta(rnet_beta),
        _tau_k(tau_k),
        _tau(tau),
        _l0_min_distance(l0_min_distance),
        _search_nn_qs(search_nn_qs),
        _ul_select_nbrs_qs(ul_select_nbrs_qs),
        _bl_select_nbrs_qs(bl_select_nbrs_qs),
        _ul_max_nbr_size(ul_max_nbr_size),
        _bl_max_nbr_size(bl_max_nbr_size)
    {
        if (!std::isfinite(rnet_beta) || rnet_beta <= ratio_t(1)) {
            ARTEA_ERROR(fmt::format("rnet_beta ({}) must be finite and > 1", rnet_beta));
        }
        if (!std::isfinite(tau_k) || tau_k < ratio_t(0)) {
            ARTEA_ERROR(fmt::format("tau_k ({}) must be finite and >= 0", tau_k));
        }
        if (!std::isfinite(tau) || tau < ratio_t(0)) {
            ARTEA_ERROR(fmt::format("shifted_coeffs / tau ({}) must be finite and >= 0", tau));
        }
        if (!std::isfinite(l0_min_distance) || l0_min_distance <= distance_t(0)) {
            ARTEA_ERROR(fmt::format("l0_min_distance ({}) must be finite and > 0", l0_min_distance));
        }
        const double l1_radius = static_cast<double>(l0_min_distance)
            * (1.0 + static_cast<double>(tau_k));
        if (!std::isfinite(l1_radius) || l1_radius > std::numeric_limits<distance_t>::max()) {
            ARTEA_ERROR(fmt::format("tau_k ({}) produces an unrepresentable L1 radius", tau_k));
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
    __attribute__((always_inline)) auto tau_k()              const -> ratio_t      { return _tau_k; }
    __attribute__((always_inline)) auto tau()                const -> ratio_t      { return _tau; }
    __attribute__((always_inline)) auto l0_min_distance()    const -> distance_t   { return _l0_min_distance; }
    __attribute__((always_inline)) auto search_nn_qs()       const -> vertex_num_t { return _search_nn_qs; }
    __attribute__((always_inline)) auto ul_select_nbrs_qs()  const -> vertex_num_t { return _ul_select_nbrs_qs; }
    __attribute__((always_inline)) auto bl_select_nbrs_qs()  const -> vertex_num_t { return _bl_select_nbrs_qs; }
    __attribute__((always_inline)) auto ul_max_nbr_size()    const -> vertex_num_t { return _ul_max_nbr_size; }
    __attribute__((always_inline)) auto bl_max_nbr_size()    const -> vertex_num_t { return _bl_max_nbr_size; }

    /**
     * @brief Covering radius for upper layer @p h (h >= 1):
     *        R_h = l0_min_distance * (1 + tau_k) * rnet_beta^(h - 1).
     *        For h == 0, return the L0 distance scale; L0 is not an r-net.
     */
    __attribute__((always_inline))
    auto radius_at(const layer_num_t h) const -> distance_t {
        if (h == 0) return _l0_min_distance;
        return static_cast<distance_t>(
            static_cast<double>(_l0_min_distance)
                * (1.0 + static_cast<double>(_tau_k))
                * std::pow(static_cast<double>(_rnet_beta), static_cast<double>(h - 1))
        );
    }

    /**
     * @brief Return the dimensionless ARC threshold Delta_h for layer h.
     *
     * gamma = (alpha + 1) / (alpha - 1) + theta
     * psi_h = (tau * rho + sum_{i=0}^{h} R_i) / R_h
     * Delta_h = (gamma + 1) * psi_h + gamma * R_{h+1} / R_h
     *
     * alpha, tau and rho come from the pruning config's scale_coeffs,
     * shifted_coeffs and l0_min_distance, respectively. R_i comes from
     * radius_at(i), including its distinct L0/L1 rule.
     * The routing slack theta is fixed locally at 1.0.
     * The corresponding distance cutoff is Delta_h * R_h.
     *
     * @param h Zero-based layer ID.
     * @param pruning_config Pruning policy providing scale_coeffs(),
     *        shifted_coeffs() and l0_min_distance(). Values are read at call
     *        time, preserving the policy's shift and distance scale even
     *        when they differ from this r-net configuration.
     * @throws std::invalid_argument If alpha <= 1 or the scalar parameters
     *         are non-finite or outside their mathematical domains.
     * @throws std::out_of_range If h + 1 cannot be represented.
     * @throws std::overflow_error If a radius or the result is not representable.
     */
    template <typename PruningConfigT>
    auto aspect_ratio_constraint(
        const layer_num_t h,
        const PruningConfigT& pruning_config
    ) const -> ratio_t {
        const auto bounds = _routing_bounds(h, pruning_config);
        const double next_radius = radius_at(h + 1);
        if (!std::isfinite(next_radius) || next_radius <= 0.0) {
            throw std::overflow_error("ARC requires finite positive layer radii");
        }

        const double psi = bounds.distance_bound / bounds.radius;
        const double arc = (bounds.gamma + 1.0) * psi
            + bounds.gamma * (next_radius / bounds.radius);
        if (!std::isfinite(arc) || arc > std::numeric_limits<ratio_t>::max()) {
            throw std::overflow_error("ARC threshold is not representable");
        }
        return static_cast<ratio_t>(arc);
    }

    /**
     * @brief Absolute distance threshold for early descent from an upper layer.
     *
     * Section 4 of main.pdf stops routing at level h >= 1 once
     * delta(v, q) <= gamma * psi_h * R_h, then descends to level h - 1.
     * Using the same radius-sum definition as aspect_ratio_constraint:
     *
     * T_h = gamma * (tau * rho + sum_{i=0}^{h} R_i)
     * gamma = (alpha + 1) / (alpha - 1) + theta, with local theta = 1.0.
     *
     * This is a distance in the units of radius_at(), not a dimensionless
     * ratio. Radii follow this configuration's L0/L1 rule. alpha, tau and
     * rho come from the supplied pruning config, just as for ARC.
     * The method only computes the threshold; routers must apply it explicitly.
     * The bottom layer must continue its final search and has no such threshold.
     *
     * @param h Upper-layer ID (h >= 1, excluding the invalid max-ID sentinel).
     * @param pruning_config Policy providing scale_coeffs(), shifted_coeffs()
     *        and l0_min_distance().
     * @throws std::out_of_range If h is zero or the invalid max-ID sentinel.
     * @throws std::invalid_argument If alpha <= 1, tau < 0, rho <= 0 or any
     *         of these values is non-finite.
     * @throws std::overflow_error If a required radius or threshold is unrepresentable.
     */
    template <typename PruningConfigT>
    auto early_stop_threshold(
        const layer_num_t h,
        const PruningConfigT& pruning_config
    ) const -> distance_t {
        if (h == 0) {
            throw std::out_of_range("Early stopping applies only to upper layers (h >= 1)");
        }
        const auto bounds = _routing_bounds(h, pruning_config);
        const double threshold = bounds.gamma * bounds.distance_bound;
        if (!std::isfinite(threshold) || threshold > std::numeric_limits<distance_t>::max()) {
            throw std::overflow_error("Early-stop threshold is not representable");
        }
        return static_cast<distance_t>(threshold);
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
     * @brief Compute the maximum allowed level ID for a dataset of @p total_vertices.
     */
    static auto compute_max_allowed_level_id(const vertex_num_t total_vertices) -> layer_num_t
    {
        if (total_vertices == 0) return 1;
        const double ratio = static_cast<double>(total_vertices) / 1000.0;
        if (ratio <= 1.0) return 1;
        const double raw = std::log(ratio);
        const layer_num_t ceiled = static_cast<layer_num_t>(std::ceil(raw));
        return std::max<layer_num_t>(ceiled, layer_num_t(1));
    }

private:
    struct RoutingBounds {
        double gamma;
        double radius;
        double distance_bound; // psi_h * R_h
    };

    /** @brief Common factors for ARC pruning and upper-layer early stopping. */
    template <typename PruningConfigT>
    auto _routing_bounds(const layer_num_t h, const PruningConfigT& pruning_config) const
        -> RoutingBounds {
        constexpr double theta = 1.0;
        const double alpha = pruning_config.scale_coeffs();
        const double tau = pruning_config.shifted_coeffs();
        const double rho = pruning_config.l0_min_distance();
        if (!std::isfinite(alpha) || alpha <= 1.0 ||
            !std::isfinite(tau) || tau < 0.0 ||
            !std::isfinite(rho) || rho <= 0.0) {
            throw std::invalid_argument(
                "Routing thresholds require finite alpha > 1, tau >= 0 and rho > 0");
        }
        if (h == std::numeric_limits<layer_num_t>::max()) {
            throw std::out_of_range("Routing thresholds require a valid layer ID");
        }
        const double radius = radius_at(h);
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::overflow_error("Routing thresholds require finite positive layer radii");
        }
        double radius_sum = radius;
        for (layer_num_t i = 0; i < h; ++i) {
            radius_sum += static_cast<double>(radius_at(i));
        }
        const double gamma = (alpha + 1.0) / (alpha - 1.0) + theta;
        return {gamma, radius, tau * rho + radius_sum};
    }

    /** @brief Radius growth factor from L1 upward. Must be > 1. */
    ratio_t      _rnet_beta;

    /** @brief Nonnegative radius coefficient: every upper-layer radius is scaled by 1 + tau_k. */
    ratio_t      _tau_k;

    /** @brief Nonnegative shift coefficient from shifted_coeffs; independent of radii. */
    ratio_t      _tau;

    /** @brief Characteristic L0 minimum distance in build-distance units. Must be > 0. */
    distance_t   _l0_min_distance;

    /** @brief Beam-search queue size for Phase 1 top-down descent. */
    vertex_num_t _search_nn_qs;

    /** @brief Beam-search queue size for Phase 2 candidate gathering at upper levels (L1+). */
    vertex_num_t _ul_select_nbrs_qs;

    /** @brief Beam-search queue size for Phase 2 candidate gathering at the bottom level (L0). */
    vertex_num_t _bl_select_nbrs_qs;

    /** @brief Per-vertex neighbor capacity at every upper layer. */
    vertex_num_t _ul_max_nbr_size;

    /** @brief Per-vertex neighbor capacity at the bottom layer (L0).
     *         Independent of @c _ul_max_nbr_size. */
    vertex_num_t _bl_max_nbr_size;
};

/** @brief stacked_rgraph reuses conv_graph's PruningConfig — the
 *         insertion path reads scale_coeffs only; shifted_coeffs is a
 *         post-refinement concern and is ignored on the insertion path. */
template <typename IndexTraitsT>
using PruningConfig = conv_graph::PruningConfig<IndexTraitsT>;

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
