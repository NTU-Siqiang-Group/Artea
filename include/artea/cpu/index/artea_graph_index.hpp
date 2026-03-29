/*
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph_index.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Artea hierarchical graph index structure extending HierarchicalGraph via CRTP.
 */

#pragma once

#include <variant>

namespace artea {
namespace cpu {
namespace artea_graph {

/**
 * @brief Artea hierarchical graph index, extending HierarchicalGraph with
 *        pruning, propagation, and vertices builder configs.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class GraphIndex :
    public IndexTraitsT::template hierarchical_graph_t<GraphIndex<IndexTraitsT>>
{
    using base_t = typename IndexTraitsT::template hierarchical_graph_t<GraphIndex<IndexTraitsT>>;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::artea_graph::propagate_config_t;
    using pruning_config_t = typename IndexTraitsT::artea_graph::pruning_config_t;
    using greedy_vertices_builder_config_t = typename IndexTraitsT::greedy_vertices_builder_config_t;
    using random_vertices_builder_config_t = typename IndexTraitsT::random_vertices_builder_config_t;
    using vertices_builder_config_t = typename IndexTraitsT::vertices_builder_config_t;

public:
    /**
     * @brief Construct a new Artea hierarchical graph index.
     * @param base_vecs Reference to the base layer vector data.
     * @param bottom_layer_config Configuration for bottom layer.
     * @param upper_layer_config Configuration for upper layers.
     * @param bottom_pruning_config Pruning configuration for bottom layer.
     * @param upper_pruning_config Pruning configuration for upper layers.
     * @param propagate_config Propagation configuration (shared between layers).
     * @param vertices_builder_config Configuration for vertices builder (greedy or random).
     */
    GraphIndex(
        const vector_array_t& base_vecs,
        const layer_config_t bottom_layer_config,
        const layer_config_t upper_layer_config,
        const pruning_config_t bottom_pruning_config,
        const pruning_config_t upper_pruning_config,
        const propagate_config_t propagate_config,
        const vertices_builder_config_t vertices_builder_config
    ) : base_t(base_vecs, bottom_layer_config, upper_layer_config),
        _bottom_pruning_config(bottom_pruning_config),
        _upper_pruning_config(upper_pruning_config),
        _propagate_config(propagate_config),
        _vertices_builder_config(vertices_builder_config)
    {}

    GraphIndex(const GraphIndex&) = delete;
    GraphIndex& operator=(const GraphIndex&) = delete;

    GraphIndex(GraphIndex&&) noexcept = default;
    GraphIndex& operator=(GraphIndex&&) noexcept = default;

    // --- Config accessors ---

    __attribute__((always_inline))
    auto bottom_pruning_config() const -> const pruning_config_t& { return _bottom_pruning_config; }

    __attribute__((always_inline))
    auto bottom_pruning_config() -> pruning_config_t& { return _bottom_pruning_config; }

    __attribute__((always_inline))
    auto upper_pruning_config() const -> const pruning_config_t& { return _upper_pruning_config; }

    __attribute__((always_inline))
    auto upper_pruning_config() -> pruning_config_t& { return _upper_pruning_config; }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto vertices_builder_config() const -> const vertices_builder_config_t& { return _vertices_builder_config; }

    __attribute__((always_inline))
    auto vertices_builder_config() -> vertices_builder_config_t& { return _vertices_builder_config; }

private:
    /** @brief Pruning configuration for bottom layer. */
    pruning_config_t _bottom_pruning_config;

    /** @brief Pruning configuration for upper layers. */
    pruning_config_t _upper_pruning_config;

    /** @brief Propagation configuration (shared between layers). */
    propagate_config_t _propagate_config;

    /** @brief Vertices builder configuration (greedy or random). */
    vertices_builder_config_t _vertices_builder_config;

};  // class GraphIndex

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
