/*
 * @FilePath: /Artea/include/artea/cpu/index/hierarchical_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: CRTP base hierarchical graph structure for multi-layer index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <utility>

namespace artea {
namespace cpu {

/**
 * @brief CRTP base hierarchical graph structure storing graph topology and layer configs.
 * @tparam IndexTraitsT The index traits type.
 * @tparam DerivedClassT The concrete derived graph type (CRTP).
 *
 * @note Layer ID mapping:
 *   - layer_id 0 is the bottom layer, stored at _layer_graphs[0]
 *   - Higher layer_id values represent upper layers
 */
template <
    typename IndexTraitsT,
    typename DerivedClassT,
    typename LayerGraphT
>
class HierarchicalGraph {

protected:
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using inter_layer_links_t = typename IndexTraitsT::inter_layer_links_t;
    using hierarchical_vecs_manager_t = typename IndexTraitsT::hierarchical_vecs_manager_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;

public:
    /** @brief The flat graph type used for each layer. */
    using layer_graph_t = LayerGraphT;

    /**
     * @brief Construct a new Hierarchical Graph object.
     * @param base_vecs Reference to the base layer vector data.
     * @param bottom_layer_config Configuration for bottom layer.
     * @param upper_layer_config Configuration for upper layers.
     */
    HierarchicalGraph(
        const vector_array_t& base_vecs,
        const layer_config_t bottom_layer_config,
        const layer_config_t upper_layer_config
    ) : _num_vertices(base_vecs.get_num_vecs()),
        _bottom_layer_config(bottom_layer_config),
        _upper_layer_config(upper_layer_config),
        _hier_vecs_manager(base_vecs),
        _inter_layer_links(_num_vertices)
    {}

    HierarchicalGraph(const HierarchicalGraph&) = delete;
    HierarchicalGraph& operator=(const HierarchicalGraph&) = delete;

    HierarchicalGraph(HierarchicalGraph&&) noexcept = default;
    HierarchicalGraph& operator=(HierarchicalGraph&&) noexcept = default;

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t { return _num_vertices; }

    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t { return _hier_vecs_manager.get_num_layers(); }

    /**
     * @brief Resize the layer graphs vector to accommodate a specific number of layers.
     * @param num_layers The total number of layers (including bottom layer).
     */
    __attribute__((always_inline))
    auto resize(const layer_id_t num_layers) -> void { _layer_graphs.resize(num_layers); }

    __attribute__((always_inline))
    auto bottom_layer_config() const -> const layer_config_t& { return _bottom_layer_config; }

    __attribute__((always_inline))
    auto bottom_layer_config() -> layer_config_t& { return _bottom_layer_config; }

    __attribute__((always_inline))
    auto upper_layer_config() const -> const layer_config_t& { return _upper_layer_config; }

    __attribute__((always_inline))
    auto upper_layer_config() -> layer_config_t& { return _upper_layer_config; }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<LayerGraphT>>& { return _layer_graphs; }

    __attribute__((always_inline))
    auto get_layer_graphs() const -> const std::vector<std::unique_ptr<LayerGraphT>>& { return _layer_graphs; }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() -> LayerGraphT& { return *_layer_graphs[0]; }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() const -> const LayerGraphT& { return *_layer_graphs[0]; }

    /**
     * @brief Get the flat graph at a given layer_id.
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) -> LayerGraphT& { return *_layer_graphs[layer_id]; }

    /**
     * @brief Get the flat graph at a given layer_id (const version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) const -> const LayerGraphT& { return *_layer_graphs[layer_id]; }

    /**
     * @brief Set the flat graph at a given layer_id.
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Unique pointer to the flat graph to set.
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id, std::unique_ptr<LayerGraphT> layer_graph) -> void {
        _layer_graphs[layer_id] = std::move(layer_graph);
    }

    /**
     * @brief Set the flat graph at a given layer_id (rvalue reference version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Reference to the flat graph to set (will be moved).
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id, LayerGraphT&& layer_graph) -> void {
        _layer_graphs[layer_id] = std::make_unique<LayerGraphT>(std::move(layer_graph));
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& { return _hier_vecs_manager.get_base_vecs(); }

    __attribute__((always_inline))
    auto get_base_vecs() const -> const vector_array_t& { return _hier_vecs_manager.get_base_vecs(); }

    __attribute__((always_inline))
    auto get_hier_vecs_manager() const -> const hierarchical_vecs_manager_t& { return _hier_vecs_manager; }

    __attribute__((always_inline))
    auto get_hier_vecs_manager() -> hierarchical_vecs_manager_t& { return _hier_vecs_manager; }

    __attribute__((always_inline))
    auto get_inter_layer_links() -> inter_layer_links_t& { return _inter_layer_links; }

    __attribute__((always_inline))
    auto get_inter_layer_links() const -> const inter_layer_links_t& { return _inter_layer_links; }

    __attribute__((always_inline))
    auto get_entry_point() const -> vertex_id_t { return _entry_point; }

    __attribute__((always_inline))
    auto set_entry_point(const vertex_id_t entry_point) -> void { _entry_point = entry_point; }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Configuration for bottom layer. */
    layer_config_t _bottom_layer_config;

    /** @brief Configuration for upper layers. */
    layer_config_t _upper_layer_config;

    /** @brief Hierarchical vector manager. */
    hierarchical_vecs_manager_t _hier_vecs_manager;

    /** @brief Flat graphs for all layers. layer_id 0 is the bottom layer at _layer_graphs[0]. */
    std::vector<std::unique_ptr<LayerGraphT>> _layer_graphs;

    /** @brief Links vertex between two adjacent layers. */
    inter_layer_links_t _inter_layer_links;

    /** @brief Entry point vertex ID for hierarchical search. */
    vertex_id_t _entry_point;

};  // class HierarchicalGraph

}   // namespace cpu
}   // namespace artea
