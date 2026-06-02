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
 * Carries the usual RNG scale/shift coefficients:
 *   - @c scale_coeffs is applied uniformly at every layer, both in the
 *     stacked-rgraph insertion path and the per-layer refinement
 *     pipeline.
 *   - @c shifted_coeffs is applied only at L0 during refinement. Upper
 *     layers (level_id > 0) force the effective shift to 0; stacked-
 *     rgraph insertion never consumes the shift regardless of layer.
 *     See @c artea_graph::IndexFactory::refine_layer for the gate.
 *   - @c l0_min_distance scales the L0 shift term: refine_layer feeds
 *     @c shifted_coeffs * @c l0_min_distance to the triangle / pruning
 *     updaters as the bare shift, so a unit-less @c shifted_coeffs grid
 *     stays comparable across datasets whose L0 distance scales differ
 *     by orders of magnitude (e.g. sift-1m raw vs gist-1m normalized).
 *     Distinct from @c rgraph_config.l0_rnet_radius, which sets the L0
 *     covering radius for r-net construction.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
struct PruningConfig {
    using ratio_t = typename IndexTraitsT::ratio_t;

    /**
     * @param scale_coeffs            RNG scale coefficient (>= 1).
     *                                Applied uniformly at every layer.
     * @param shifted_coeffs          RNG shift coefficient (>= 0).
     *                                Applied only at L0 during refinement;
     *                                upper layers force effective shift to 0.
     * @param l0_min_distance         Per-dataset characteristic L0 distance
     *                                that scales the shift term in
     *                                @c refine_layer. Default 1 keeps the
     *                                shift bare (no scaling). Pass the
     *                                approximate min L0 pairwise distance
     *                                (NOT @c l0_rnet_radius — that is
     *                                geometrically the covering radius and
     *                                serves a different purpose).
     */
    PruningConfig(
        ratio_t scale_coeffs,
        ratio_t shifted_coeffs,
        ratio_t l0_min_distance         = ratio_t(1)
    ) :
        _scale_coeffs(scale_coeffs),
        _shifted_coeffs(shifted_coeffs),
        _l0_min_distance(l0_min_distance)
    {}

    // Builder-style setters (chainable)
    auto scale_coeffs(ratio_t v)            -> PruningConfig& { _scale_coeffs = v;            return *this; }
    auto shifted_coeffs(ratio_t v)          -> PruningConfig& { _shifted_coeffs = v;          return *this; }
    auto l0_min_distance(ratio_t v)         -> PruningConfig& { _l0_min_distance = v;         return *this; }

    // Const getters
    auto scale_coeffs()            const -> ratio_t { return _scale_coeffs; }
    auto shifted_coeffs()          const -> ratio_t { return _shifted_coeffs; }
    auto l0_min_distance()         const -> ratio_t { return _l0_min_distance; }

    /**
     * @brief Project down to a conv_graph::PruningConfig (scale + shift
     *        only). Used by @c artea_graph::IndexStructure to forward
     *        the RNG subset to the stacked_rgraph base constructor,
     *        which stores its own scale/shift-only config. The L0-only
     *        @c l0_min_distance scaling is intentionally dropped here:
     *        the stacked-rgraph insertion path doesn't consume the
     *        shift anyway, and conv_graph callers (used in non-
     *        hierarchical contexts) have no notion of an L0 distance.
     */
    auto to_rng_only() const -> conv_graph::PruningConfig<IndexTraitsT> {
        return conv_graph::PruningConfig<IndexTraitsT>(_scale_coeffs, _shifted_coeffs);
    }

private:
    ratio_t _scale_coeffs;
    ratio_t _shifted_coeffs;
    ratio_t _l0_min_distance;
};

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
