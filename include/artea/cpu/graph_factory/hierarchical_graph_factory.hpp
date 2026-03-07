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

#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT, typename DerivedClassT>
class HierarchicalGraphFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    // Using propagate_engine_t with no selective scheduling currently.
    using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<false>;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;

public:
    HierarchicalGraphFactory() = default;

    template <typename... Args>
    auto construct_graph(
        const vector_dataset_t& dataset,
        const vertex_num_t bl_max_nbr_size,
        const vertex_num_t ul_max_nbr_size,
        const vertex_num_t bl_reserved_nbr_size,
        const vertex_num_t ul_reserved_nbr_size,
        Args&&... args
    ) -> hierarchical_graph_t {
        return static_cast<DerivedClassT*>(this)->construct_graph_impl(
            dataset,
            bl_max_nbr_size,
            ul_max_nbr_size,
            bl_reserved_nbr_size,
            ul_reserved_nbr_size,
            std::forward<Args>(args)...
        );
    }

};  // class HierarchicalGraphFactory

}   // namespace cpu
}   // namespace artea