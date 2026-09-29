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

#include <memory>
#include <optional>
#include <utility>

#include <artea/cpu/index/exact_artea/configs.hpp>
#include <artea/cpu/index/stacked_rgraph/index_structure.hpp>

namespace artea {
namespace cpu {
namespace exact_artea {

template <typename GraphFactoryTraitsT> class RNetsFactory;

/** @brief Index scaffold with stacked-rgraph storage and ARTEA configuration. */
template <typename IndexTraitsT>
class IndexStructure : public stacked_rgraph::IndexStructure<IndexTraitsT> {
    using base_t = stacked_rgraph::IndexStructure<IndexTraitsT>;

public:
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using ratio_t = typename IndexTraitsT::ratio_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using rgraph_config_t = typename IndexTraitsT::exact_artea::rgraph_config_t;
    using pruning_config_t = typename IndexTraitsT::exact_artea::pruning_config_t;

    IndexStructure(
        const vertex_num_t total_vertices,
        const rgraph_config_t& rgraph_config,
        pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0))
    ) : base_t(total_vertices, rgraph_config, pruning_config.to_rng_only()),
        _pruning_config(pruning_config) {}

    /** @brief Own a dataset; call prepare_build before constructing the graph. */
    explicit IndexStructure(std::unique_ptr<vector_dataset_t> dataset)
        : base_t(std::move(dataset)) {}

    IndexStructure(
        std::unique_ptr<vector_dataset_t> dataset,
        const rgraph_config_t& rgraph_config,
        pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0))
    ) : base_t(std::move(dataset)) {
        prepare_build(rgraph_config, pruning_config);
    }

    /** @brief Borrow immutable vectors that must outlive this index. */
    IndexStructure(
        const vector_array_t& vectors,
        const rgraph_config_t& rgraph_config,
        pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0))
    ) : base_t(vectors, rgraph_config, pruning_config.to_rng_only()),
        _pruning_config(pruning_config) {}

    IndexStructure(vector_array_t&&, const rgraph_config_t&,
                   pruning_config_t = pruning_config_t(ratio_t(1.1), ratio_t(0))) = delete;
    IndexStructure(const vector_array_t&&, const rgraph_config_t&,
                   pruning_config_t = pruning_config_t(ratio_t(1.1), ratio_t(0))) = delete;

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&) = delete;
    IndexStructure& operator=(IndexStructure&&) = delete;

    /** @brief Forward graph setup while retaining the complete ARTEA pruning config. */
    auto prepare_build(
        const rgraph_config_t& rgraph_config,
        pruning_config_t pruning_config = pruning_config_t(ratio_t(1.1), ratio_t(0))
    ) -> IndexStructure& {
        base_t::prepare_build(rgraph_config, pruning_config.to_rng_only());
        _pruning_config = pruning_config;
        return *this;
    }

    auto pruning_config() const -> const pruning_config_t& {
        return _pruning_config.value();
    }

private:
    // Bulk construction publishes a graph only after all layer assignments succeed.
    template <typename GraphFactoryTraitsT> friend class RNetsFactory;

    // The base stores the RNG-only projection needed by stacked-rgraph insertion.
    std::optional<pruning_config_t> _pruning_config;

    // TODO: Add exact_artea-specific state when its algorithm is implemented.
};

}   // namespace exact_artea
}   // namespace cpu
}   // namespace artea
