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

#include <artea/cpu/index/exact_artea/index_structure.hpp>
#include <artea/cpu/index/stacked_rgraph/index_factory.hpp>

namespace artea {
namespace cpu {
namespace exact_artea {

/** @brief Exact ARTEA factory with stacked-rgraph construction primitives. */
template <typename GraphFactoryTraitsT>
class IndexFactory : public stacked_rgraph::IndexFactory<GraphFactoryTraitsT> {
    // TODO: Implement exact_artea edge construction and refinement.
};

}   // namespace exact_artea
}   // namespace cpu
}   // namespace artea
