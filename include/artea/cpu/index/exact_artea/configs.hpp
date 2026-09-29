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

#pragma once

#include <artea/cpu/index/artea_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace exact_artea {

/** @brief Reuse ARTEA's r-net configuration. */
template <typename IndexTraitsT>
using RGraphConfig = artea_graph::RGraphConfig<IndexTraitsT>;

/** @brief Reuse ARTEA's pruning configuration, including l0_min_distance. */
template <typename IndexTraitsT>
using PruningConfig = artea_graph::PruningConfig<IndexTraitsT>;

}   // namespace exact_artea
}   // namespace cpu
}   // namespace artea
