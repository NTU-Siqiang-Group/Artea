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
 * @FilePath: /Artea/include/artea/cpu/configs/pruning_config.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for pruning during edge generation.
 */

#pragma once

namespace artea {
namespace cpu {

namespace conv_graph {

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

namespace artea_graph {

/** @brief Artea graph uses the same PruningConfig as conv_graph. */
template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

}   // namespace artea_graph

namespace knn_graph {

/** @brief KNN graph uses the same PruningConfig as conv_graph. */
template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

}   // namespace knn_graph

}   // namespace cpu
}   // namespace artea
