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
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using edges_builder_config_t = typename GraphFactoryTraitsT::artea_graph::edges_builder_config_t;
    using vg_policy_t = typename GraphFactoryTraitsT::vg_policy_t;
    using eg_policy_t = typename GraphFactoryTraitsT::eg_policy_t;

public:
    HierarchicalGraphFactory() = default;

    template <vg_policy_t VGPolicy, eg_policy_t EGPolicy, typename... Args>
    auto construct_graph(
        const vector_dataset_t& dataset,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        edges_builder_config_t bottom_edges_builder_config,
        edges_builder_config_t upper_edges_builder_config,
        Args&&... args
    ) -> hierarchical_graph_t {
        return template construct_graph<VGPolicy, EGPolicy>(
            dataset.get_base_vecs(),
            bottom_layer_config,
            upper_layer_config,
            bottom_edges_builder_config,
            upper_edges_builder_config,
            std::forward<Args>(args)...
        );
    }

    template <vg_policy_t VGPolicy, eg_policy_t EGPolicy, typename... Args>
    auto construct_graph(
        const vector_array_t& base_vecs,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        edges_builder_config_t bottom_edges_builder_config,
        edges_builder_config_t upper_edges_builder_config,
        Args&&... args
    ) -> hierarchical_graph_t {
        return static_cast<DerivedClassT*>(this)->template construct_graph_impl<VGPolicy, EGPolicy>(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_edges_builder_config,
            upper_edges_builder_config,
            std::forward<Args>(args)...
        );
    }

};  // class HierarchicalGraphFactory

}   // namespace cpu
}   // namespace artea