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
 * @FilePath: /Artea/include/artea/cpu/graph_factory/descent_config.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for convergent graph descent algorithm.
 */

#pragma once

namespace artea {
namespace cpu {

/**
 * @brief Configuration for convergent graph descent algorithm.
 * Supports builder pattern for flexible configuration.
 * @tparam GraphFactoryTraitsT The graph factory traits type.
 */
template <typename GraphFactoryTraitsT>
struct DescentConfig {
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;

    /**
     * @brief Constructor for descent configuration.
     * @param scale_coeffs Scale coefficient for RNG pruning.
     * @param shifted_coeffs Shift coefficient for RNG pruning.
     * @param num_outer_iters Number of outer iterations (recommend: 4).
     * @param num_inner_iters Number of inner iterations (recommend: 14).
     */
    DescentConfig(
        ratio_t scale_coeffs,
        ratio_t shifted_coeffs,
        iter_t num_outer_iters,
        iter_t num_inner_iters
    ) :
        _scale_coeffs(scale_coeffs),
        _shifted_coeffs(shifted_coeffs),
        _num_outer_iters(num_outer_iters),
        _num_inner_iters(num_inner_iters)
    {}

    // Builder pattern setters (chainable)
    auto scale_coeffs(ratio_t value) -> DescentConfig& { _scale_coeffs = value; return *this; }
    auto shifted_coeffs(ratio_t value) -> DescentConfig& { _shifted_coeffs = value; return *this; }
    auto num_outer_iters(iter_t value) -> DescentConfig& { _num_outer_iters = value; return *this; }
    auto num_inner_iters(iter_t value) -> DescentConfig& { _num_inner_iters = value; return *this; }

    // Const getters
    auto scale_coeffs() const -> ratio_t { return _scale_coeffs; }
    auto shifted_coeffs() const -> ratio_t { return _shifted_coeffs; }
    auto num_outer_iters() const -> iter_t { return _num_outer_iters; }
    auto num_inner_iters() const -> iter_t { return _num_inner_iters; }

private:
    /** @brief Scale coefficient for RNG pruning. */
    ratio_t _scale_coeffs;

    /** @brief Shift coefficient for RNG pruning. */
    ratio_t _shifted_coeffs;

    /** @brief Number of outer iterations (recommend: 4). */
    iter_t _num_outer_iters;

    /** @brief Number of inner iterations (recommend: 14). */
    iter_t _num_inner_iters;
};

}   // namespace cpu
}   // namespace artea
