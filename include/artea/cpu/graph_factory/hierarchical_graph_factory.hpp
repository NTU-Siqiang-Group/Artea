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

    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using pruning_config_t = typename GraphFactoryTraitsT::artea_graph::pruning_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::artea_graph::propagate_config_t;
    using greedy_vertices_builder_config_t = typename GraphFactoryTraitsT::greedy_vertices_builder_config_t;

public:
    HierarchicalGraphFactory() = default;

    static auto construct_graph(
        const vector_dataset_t& dataset,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        pruning_config_t bottom_pruning_config,
        pruning_config_t upper_pruning_config,
        propagate_config_t propagate_config,
        greedy_vertices_builder_config_t vertices_builder_config
    ) -> hierarchical_graph_t {
        return DerivedClassT::construct_graph_impl(
            dataset.get_base_vecs(),
            bottom_layer_config,
            upper_layer_config,
            bottom_pruning_config,
            upper_pruning_config,
            propagate_config,
            vertices_builder_config
        );
    }

    static auto construct_graph(
        const vector_array_t& base_vecs,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        pruning_config_t bottom_pruning_config,
        pruning_config_t upper_pruning_config,
        propagate_config_t propagate_config,
        greedy_vertices_builder_config_t vertices_builder_config
    ) -> hierarchical_graph_t {
        return DerivedClassT::construct_graph_impl(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_pruning_config,
            upper_pruning_config,
            propagate_config,
            vertices_builder_config
        );
    }

};  // class HierarchicalGraphFactory

}   // namespace cpu
}   // namespace artea
