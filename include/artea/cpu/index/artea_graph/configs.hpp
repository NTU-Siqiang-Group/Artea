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
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configs for artea_graph. Reuses conv_graph's PropagateConfig
 *               and stacked_rgraph's RGraphConfig. PruningConfig is standalone:
 *               on top of the usual scale/shift it carries an
 *               aspect-ratio-constrained (ARC) pruning policy consumed
 *               as a final sweep in refine_layer.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>
#include <artea/cpu/index/stacked_rgraph/configs.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/** @brief artea_graph reuses conv_graph's propagate config. */
template <typename IndexTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<IndexTraitsT>;

/** @brief artea_graph reuses stacked_rgraph's r-net geometry config. */
template <typename IndexTraitsT>
using RGraphConfig = stacked_rgraph::RGraphConfig<IndexTraitsT>;

/**
 * @brief Pruning config for artea_graph.
 *
 * Carries the usual RNG scale/shift coefficients (consumed by the
 * stacked-rgraph upper-layer insertion path — reads @c scale_coeffs
 * only — and the per-layer refinement @c TriangleUpdater /
 * @c PruningUpdater which read both), plus an aspect-ratio-constrained
 * (ARC) pruning toggle applied as a final sweep at the end of
 * @c refine_layer:
 *   - When @c perform_arc is true, every edge longer than
 *     @c aspect_ratio_constraint * @c radius_at(level_id) is dropped
 *     at layer @p level_id.
 *   - When @c perform_arc is false, the ARC sweep is skipped entirely.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
struct PruningConfig {
    using ratio_t = typename IndexTraitsT::ratio_t;

    /**
     * @param scale_coeffs            RNG scale coefficient (>= 1).
     * @param shifted_coeffs          RNG shift coefficient (>= 0).
     * @param perform_arc             Whether refine_layer runs the final
     *                                ARC sweep. Default false.
     * @param aspect_ratio_constraint Multiplier applied to the per-layer
     *                                r-net radius to obtain the ARC
     *                                threshold. Only consumed when
     *                                @p perform_arc is true. Default 1.0
     *                                (drop edges longer than the layer's
     *                                covering radius).
     */
    PruningConfig(
        ratio_t scale_coeffs,
        ratio_t shifted_coeffs,
        bool    perform_arc             = false,
        ratio_t aspect_ratio_constraint = ratio_t(1)
    ) :
        _scale_coeffs(scale_coeffs),
        _shifted_coeffs(shifted_coeffs),
        _perform_arc(perform_arc),
        _aspect_ratio_constraint(aspect_ratio_constraint)
    {}

    // Builder-style setters (chainable)
    auto scale_coeffs(ratio_t v)            -> PruningConfig& { _scale_coeffs = v;            return *this; }
    auto shifted_coeffs(ratio_t v)          -> PruningConfig& { _shifted_coeffs = v;          return *this; }
    auto perform_arc(bool v)                -> PruningConfig& { _perform_arc = v;             return *this; }
    auto aspect_ratio_constraint(ratio_t v) -> PruningConfig& { _aspect_ratio_constraint = v; return *this; }

    // Const getters
    auto scale_coeffs()            const -> ratio_t { return _scale_coeffs; }
    auto shifted_coeffs()          const -> ratio_t { return _shifted_coeffs; }
    auto perform_arc()             const -> bool    { return _perform_arc; }
    auto aspect_ratio_constraint() const -> ratio_t { return _aspect_ratio_constraint; }

    /**
     * @brief Project down to a conv_graph::PruningConfig (scale + shift
     *        only). Used by @c artea_graph::IndexStructure to forward
     *        the RNG subset to the stacked_rgraph base constructor,
     *        which stores its own scale/shift-only config.
     */
    auto to_rng_only() const -> conv_graph::PruningConfig<IndexTraitsT> {
        return conv_graph::PruningConfig<IndexTraitsT>(_scale_coeffs, _shifted_coeffs);
    }

private:
    ratio_t _scale_coeffs;
    ratio_t _shifted_coeffs;
    bool    _perform_arc;
    ratio_t _aspect_ratio_constraint;
};

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
