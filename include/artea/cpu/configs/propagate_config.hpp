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
 * @FilePath: /Artea/include/artea/cpu/configs/propagate_config.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for propagation during edge generation.
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

}   // namespace conv_graph

namespace artea_graph {

/** @brief Artea graph uses the same PropagateConfig as conv_graph. */
template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

}   // namespace artea_graph

}   // namespace cpu
}   // namespace artea
