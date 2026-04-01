/*
 * @FilePath: /Artea/include/artea/cpu/index/conv_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for convergent graph construction.
 */

#pragma once

namespace artea {
namespace cpu {
namespace conv_graph {

/**
 * @brief Configuration for propagation during edge generation.
 * Supports builder pattern for flexible configuration.
 * @tparam BaseTraitsT The index traits type.
 */
template <typename BaseTraitsT>
struct PropagateConfig {
    using ratio_t = typename BaseTraitsT::ratio_t;
    using iter_t = typename BaseTraitsT::iter_t;

    /**
     * @brief Constructor for propagation configuration.
     * @param num_build_loops Number of build loops (recommend: 4).
     * @param num_triu_iters Number of triangle updater iterations per build loop (recommend: 14).
     * @param prefill_ratio Prefill ratio for initial random graph (init_nbr_size = max_nbr_size * prefill_ratio).
     * @param num_routing_loops Number of routing updater iterations applied at the end of the final build loop (recommend: 1).
     */
    PropagateConfig(
        iter_t num_build_loops,
        iter_t num_triu_iters,
        ratio_t prefill_ratio = ratio_t(1),
        iter_t num_routing_loops = iter_t(1)
    ) :
        _num_build_loops(num_build_loops),
        _num_triu_iters(num_triu_iters),
        _prefill_ratio(prefill_ratio),
        _num_routing_loops(num_routing_loops)
    {}

    // Builder pattern setters (chainable)
    auto num_build_loops(iter_t value) -> PropagateConfig& { _num_build_loops = value; return *this; }
    auto num_triu_iters(iter_t value) -> PropagateConfig& { _num_triu_iters = value; return *this; }
    auto prefill_ratio(ratio_t value) -> PropagateConfig& { _prefill_ratio = value; return *this; }
    auto num_routing_loops(iter_t value) -> PropagateConfig& { _num_routing_loops = value; return *this; }

    // Const getters
    auto num_build_loops() const -> iter_t { return _num_build_loops; }
    auto num_triu_iters() const -> iter_t { return _num_triu_iters; }
    auto prefill_ratio() const -> ratio_t { return _prefill_ratio; }
    auto num_routing_loops() const -> iter_t { return _num_routing_loops; }

private:
    /** @brief Number of build loops (recommend: 4). */
    iter_t _num_build_loops;

    /** @brief Number of triangle updater iterations (recommend: 14). */
    iter_t _num_triu_iters;

    /** @brief Prefill ratio for initial random graph (init_nbr_size = max_nbr_size * prefill_ratio). */
    ratio_t _prefill_ratio;

    /** @brief Number of routing updater iterations applied at the end of the final build loop (recommend: 1). */
    iter_t _num_routing_loops;
};

/**
 * @brief Configuration for pruning during edge generation.
 * Supports builder pattern for flexible configuration.
 * @tparam BaseTraitsT The index traits type.
 */
template <typename BaseTraitsT>
struct PruningConfig {
    using ratio_t = typename BaseTraitsT::ratio_t;

    /**
     * @brief Constructor for pruning configuration.
     * @param scale_coeffs Scale coefficient for RNG pruning.
     * @param shifted_coeffs Shift coefficient for RNG pruning.
     */
    PruningConfig(ratio_t scale_coeffs, ratio_t shifted_coeffs) :
        _scale_coeffs(scale_coeffs),
        _shifted_coeffs(shifted_coeffs)
    {}

    // Builder pattern setters (chainable)
    auto scale_coeffs(ratio_t value) -> PruningConfig& { _scale_coeffs = value; return *this; }
    auto shifted_coeffs(ratio_t value) -> PruningConfig& { _shifted_coeffs = value; return *this; }

    // Const getters
    auto scale_coeffs() const -> ratio_t { return _scale_coeffs; }
    auto shifted_coeffs() const -> ratio_t { return _shifted_coeffs; }

private:
    /** @brief Scale coefficient for RNG pruning. */
    ratio_t _scale_coeffs;

    /** @brief Shift coefficient for RNG pruning. */
    ratio_t _shifted_coeffs;
};

}   // namespace conv_graph
}   // namespace cpu
}   // namespace artea
