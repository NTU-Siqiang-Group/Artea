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
class FlatGraphFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    // Using propagate_engine_t with no selective scheduling currently.
    using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<false>;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;

public:
    FlatGraphFactory() = default;

    template <typename... Args>
    auto construct_graph(
        const vector_dataset_t& dataset,
        const layer_config_t& layer_config,
        Args&&... args
    ) -> flat_graph_t {
        return construct_graph(
            dataset.get_base_vecs(),
            layer_config,
            std::forward<Args>(args)...
        );
    }

    template <typename... Args>
    auto construct_graph(
        const vector_array_t& base_vecs,
        const layer_config_t& layer_config,
        Args&&... args
    ) -> flat_graph_t {
        return static_cast<DerivedClassT*>(this)->construct_graph_impl(
            base_vecs,
            layer_config,
            std::forward<Args>(args)...
        );
    }

};  // class FlatGraphFactory

}   // namespace cpu
}   // namespace artea