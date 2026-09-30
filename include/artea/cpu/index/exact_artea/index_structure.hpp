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

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <artea/cpu/index/dataset_index.hpp>
#include <artea/cpu/index/exact_artea/configs.hpp>
#include <artea/cpu/index/layer_config.hpp>
#include <artea/cpu/index/dynamic_structure/refining_graph.hpp>
#include <artea/cpu/index/compact_structure/hierarchical_graph.hpp>

namespace artea::cpu::exact_artea {

template <typename GraphFactoryTraitsT> class RNetsFactory;
template <typename GraphFactoryTraitsT> class IndexFactory;

/**
 * @brief Independent exact index: one variable-degree RefiningGraph per layer.
 *
 * Reuses dataset ownership and ARTEA configuration, but no stacked-rgraph
 * construction or fixed-size neighbor slots. All layers borrow the same global
 * vectors. A separately owned CSR graph is published by IndexFactory::merge_layers.
 * Rebuilds and layer mutations require queries to have stopped.
 */
template <typename IndexTraitsT>
class IndexStructure : public DatasetIndex<IndexTraitsT> {
    using data_base_t = DatasetIndex<IndexTraitsT>;

public:
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using ratio_t = typename IndexTraitsT::ratio_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using rgraph_config_t = typename IndexTraitsT::exact_artea::rgraph_config_t;
    using pruning_config_t = typename IndexTraitsT::exact_artea::pruning_config_t;
    using refining_graph_t = typename IndexTraitsT::dynamic::refining_graph_t;
    using hierarchical_graph_t = typename IndexTraitsT::compact::hierarchical_graph_t;
    using compact_graph_t = hierarchical_graph_t;

    // The index itself exposes nbr_t rows from its per-layer building graphs.
    // get_hierarchical_graph() exposes the merged, topology-only CSR graph.
    static constexpr bool is_compacted = false;
    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;
    static constexpr layer_id_t invalid_level_id = hierarchical_graph_t::invalid_level_id;

    explicit IndexStructure(std::unique_ptr<vector_dataset_t> dataset)
        : data_base_t(std::move(dataset)) {}

    IndexStructure(std::unique_ptr<vector_dataset_t> dataset,
                   const rgraph_config_t& config,
                   pruning_config_t pruning = pruning_config_t(ratio_t(1.1), ratio_t(0)))
        : data_base_t(std::move(dataset)) {
        prepare_build(config, pruning);
    }

    /** @brief Borrow immutable vectors that must outlive this index and all its graphs. */
    IndexStructure(const vector_array_t& vectors,
                   const rgraph_config_t& config,
                   pruning_config_t pruning = pruning_config_t(ratio_t(1.1), ratio_t(0)))
        : data_base_t(vectors) {
        prepare_build(config, pruning);
    }

    IndexStructure(vector_array_t&&, const rgraph_config_t&,
                   pruning_config_t = pruning_config_t(ratio_t(1.1), ratio_t(0))) = delete;
    IndexStructure(const vector_array_t&&, const rgraph_config_t&,
                   pruning_config_t = pruning_config_t(ratio_t(1.1), ratio_t(0))) = delete;
    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&) = delete;
    IndexStructure& operator=(IndexStructure&&) = delete;

    /** @brief Reset construction over the same dataset, invalidating all graph references. */
    auto prepare_build(const rgraph_config_t& config,
                       pruning_config_t pruning = pruning_config_t(ratio_t(1.1), ratio_t(0)))
        -> IndexStructure& {
        _hierarchical_graph.reset();
        _layers.clear();
        _highest_levels.clear();
        _rgraph_config = config;
        _pruning_config = pruning;
        return *this;
    }

    auto rgraph_config() const -> const rgraph_config_t& {
        if (!_rgraph_config) throw std::logic_error("Exact index requires prepare_build");
        return *_rgraph_config;
    }
    auto pruning_config() const -> const pruning_config_t& {
        if (!_pruning_config) throw std::logic_error("Exact index requires prepare_build");
        return *_pruning_config;
    }
    auto radius_at(layer_id_t h) const -> distance_t { return rgraph_config().radius_at(h); }
    auto has_layers() const noexcept -> bool { return !_layers.empty(); }
    auto has_hierarchical_graph() const noexcept -> bool { return _hierarchical_graph.has_value(); }
    auto get_num_layers() const -> std::size_t { return _layers.size(); }
    auto get_num_vertices() const -> vertex_num_t {
        return static_cast<vertex_num_t>(_highest_levels.size());
    }
    auto get_highest_level_id(vertex_id_t vid) const -> layer_id_t { return _highest_levels.at(vid); }
    auto top_occupied_level_id() const -> layer_id_t {
        return _highest_levels.empty() ? invalid_level_id : static_cast<layer_id_t>(_layers.size() - 1);
    }

    auto get_layer_graph(layer_id_t h) const -> const refining_graph_t& { return *_layers.at(h); }

    /** @brief Access a mutable layer, invalidating the merged snapshot.
     * Do not retain mutable references across a subsequent merge or rebuild.
     * Topology arrays and ID mappings must remain unchanged; mutate neighbor rows only.
     */
    auto get_layer_graph(layer_id_t h) -> refining_graph_t& {
        auto& graph = *_layers.at(h);
        _hierarchical_graph.reset();
        return graph;
    }
    auto layer_vids(layer_id_t h) const -> std::span<const vertex_id_t> {
        return get_layer_graph(h).local_to_global();
    }
    auto fetch_level_nbrs(vertex_id_t vid, layer_id_t h) const -> std::span<const nbr_t> {
        if (get_highest_level_id(vid) < h) throw std::out_of_range("Vertex is not in this layer");
        return get_layer_graph(h).fetch_nbrs(vid);
    }
    auto get_top_level_vids() const -> std::span<const vertex_id_t> {
        return _highest_levels.empty() ? std::span<const vertex_id_t>{} : layer_vids(top_occupied_level_id());
    }
    auto entry_point_vid() const -> vertex_id_t {
        const auto top = get_top_level_vids();
        return top.empty() ? invalid_vertex_id : top.front();
    }
    auto get_hierarchical_graph() const -> const hierarchical_graph_t& {
        if (!_hierarchical_graph) throw std::logic_error("Exact index requires merge_layers");
        return *_hierarchical_graph;
    }
    auto get_compact_graph() const -> const compact_graph_t& { return get_hierarchical_graph(); }

private:
    template <typename GraphFactoryTraitsT> friend class RNetsFactory;
    template <typename GraphFactoryTraitsT> friend class IndexFactory;

    /** @brief Publish all layer memberships atomically after FPS succeeds. */
    auto initialize_layers(std::vector<layer_id_t> highest_levels) -> void {
        const auto& vectors = this->get_base_vecs();
        const std::size_t count = vectors.get_num_vecs();
        if (highest_levels.size() != count || count >= invalid_vertex_id) {
            throw std::length_error("Exact layer membership does not fit the dataset");
        }
        const layer_id_t top = highest_levels.empty() ? 0 :
            *std::max_element(highest_levels.begin(), highest_levels.end());
        if (top == invalid_level_id) throw std::out_of_range("Invalid exact layer height");
        std::vector<std::unique_ptr<refining_graph_t>> layers;
        layers.reserve(static_cast<std::size_t>(top) + 1);
        for (layer_id_t h = 0; h <= top; ++h) {
            std::vector<vertex_id_t> local_to_global;
            std::vector<vertex_id_t> global_to_local(count, invalid_vertex_id);
            for (vertex_id_t vid = 0; vid < count; ++vid) {
                if (highest_levels[vid] < h) continue;
                global_to_local[vid] = static_cast<vertex_id_t>(local_to_global.size());
                local_to_global.push_back(vid);
            }
            // RefiningGraph's legacy config is only an initial vector reservation.
            // Exact construction uses growing vectors and never applies a degree cap.
            layers.push_back(std::make_unique<refining_graph_t>(
                vectors, typename IndexTraitsT::layer_config_t(0),
                std::move(local_to_global), std::move(global_to_local)));
        }
        _hierarchical_graph.reset();
        _highest_levels.swap(highest_levels);
        _layers.swap(layers);
    }

    std::optional<rgraph_config_t> _rgraph_config;
    std::optional<pruning_config_t> _pruning_config;
    std::vector<layer_id_t> _highest_levels;
    std::vector<std::unique_ptr<refining_graph_t>> _layers;
    std::optional<hierarchical_graph_t> _hierarchical_graph;
};

} // namespace artea::cpu::exact_artea
