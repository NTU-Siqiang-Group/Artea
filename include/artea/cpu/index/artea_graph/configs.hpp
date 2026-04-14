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
 *               and PruningConfig verbatim — the per-layer refinement step
 *               drives a conv_graph build, so the parameter set is identical.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>
#include <artea/cpu/index/stacked_rgraph/configs.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/** @brief artea_graph reuses conv_graph's propagate/pruning configs —
 *         the per-layer refinement drives a conv_graph build. */
template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

/** @brief artea_graph reuses stacked_rgraph's r-net geometry config. */
template <typename BaseTraitsT>
using RGraphConfig = stacked_rgraph::RGraphConfig<BaseTraitsT>;

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
