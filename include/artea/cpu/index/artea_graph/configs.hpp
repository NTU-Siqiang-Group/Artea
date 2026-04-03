/*
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration aliases for Artea graph construction.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/** @brief Artea graph uses the same PropagateConfig as conv_graph. */
template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

/** @brief Artea graph uses the same PruningConfig as conv_graph. */
template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

/**
 * @brief Configuration for GraphMIS-based R-net vertex selection in Artea hierarchical graph.
 * @tparam BaseTraitsT The base traits type.
 */
template <typename BaseTraitsT>
struct RNetConfig {
    using distance_t = typename BaseTraitsT::distance_t;
    using ratio_t = typename BaseTraitsT::ratio_t;
    using vertex_num_t = typename BaseTraitsT::vertex_num_t;

    /** @brief Quantile used to probe rnet_radius from the KNN graph's nearest-neighbor distances. */
    static constexpr float target_quantile = 0.999f;

    /** @brief Neighbor rank used to probe rnet_radius (1-indexed: 1 = nearest neighbor). */
    static constexpr vertex_num_t target_rank = 1;

    /**
     * @brief Constructor for R-net configuration.
     * @param mis_radix Geometric ratio between consecutive coarse-to-fine MIS radius steps (default: 1.2).
     * @param mis_max_power Number of coarse MIS steps above rnet_radius (0 = single-shot, default: 12).
     * @param rnet_beta Radius growth factor between layers (next_radius = prev_radius * rnet_beta, default: 1.44).
     */
    RNetConfig(
        distance_t mis_radix = distance_t(1.2),
        uint32_t mis_max_power = uint32_t(12),
        ratio_t rnet_beta = ratio_t(1.44)
    ) :
        _mis_radix(mis_radix),
        _mis_max_power(mis_max_power),
        _rnet_beta(rnet_beta)
    {}

    // Builder pattern setters (chainable)
    auto mis_radix(distance_t value) -> RNetConfig& { _mis_radix = value; return *this; }
    auto mis_max_power(uint32_t value) -> RNetConfig& { _mis_max_power = value; return *this; }
    auto rnet_beta(ratio_t value) -> RNetConfig& { _rnet_beta = value; return *this; }

    // Const getters
    auto mis_radix() const -> distance_t { return _mis_radix; }
    auto mis_max_power() const -> uint32_t { return _mis_max_power; }
    auto rnet_beta() const -> ratio_t { return _rnet_beta; }

private:
    /** @brief Geometric ratio between consecutive coarse-to-fine MIS radius steps. */
    distance_t _mis_radix;

    /** @brief Number of coarse MIS steps above rnet_radius (0 = single-shot). */
    uint32_t _mis_max_power;

    /** @brief Radius growth factor between layers. */
    ratio_t _rnet_beta;
};

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
