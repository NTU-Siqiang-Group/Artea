/*
 * @FilePath: /Artea/include/artea/cpu/index/descent_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: CRTP base descent graph structure for graph-based index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <nlohmann/json.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Descent Graph: CRTP base graph structure suitable for gradient descent scenarios.
 *        Stores only graph topology and layer config.
 *        Subclasses (e.g. conv_graph::IndexStructure) extend with algorithm-specific configs.
 * @tparam IndexTraitsT The index traits type.
 * @tparam DerivedClassT The concrete derived graph type (CRTP).
 */
template <typename IndexTraitsT, typename DerivedClassT>
class DescentGraph {

protected:
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using dnbr_t = typename IndexTraitsT::dnbr_t;
    using dnbr_arr_t = typename IndexTraitsT::dnbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;

public:
    /**
     * @brief Construct a new Descent Graph object.
     * @param vecs_data Reference to the vector data for this layer.
     * @param layer_config Layer configuration (max_nbr_size and reserved_nbr_size).
     */
    DescentGraph(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config
    ) :
        _num_vertices(vecs_data.get_num_vecs()),
        _layer_config(layer_config),
        _vecs_data(vecs_data)
    {
        _nbrs_arr.resize(_num_vertices);
        for (vertex_num_t i = 0; i < _num_vertices; ++i) {
            _nbrs_arr[i].reserve(layer_config.reserved_nbr_size());
        }
    }

    DescentGraph(const DescentGraph&) = delete;
    DescentGraph& operator=(const DescentGraph&) = delete;

    DescentGraph(DescentGraph&& other) noexcept
        : _num_vertices(other._num_vertices),
          _layer_config(other._layer_config),
          _nbrs_arr(std::move(other._nbrs_arr)),
          _vecs_data(other._vecs_data)
    {}

    DescentGraph& operator=(DescentGraph&& other) noexcept {
        _num_vertices = other._num_vertices;
        _layer_config = other._layer_config;
        _nbrs_arr = std::move(other._nbrs_arr);
        // _vecs_data is a reference, cannot be reseated
        return *this;
    }

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t { return _num_vertices; }

    __attribute__((always_inline))
    auto layer_config() const -> const layer_config_t& { return _layer_config; }

    __attribute__((always_inline))
    auto layer_config() -> layer_config_t& { return _layer_config; }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<dnbr_arr_t>& { return _nbrs_arr; }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<dnbr_arr_t>& { return _nbrs_arr; }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> const dnbr_arr_t& { return _nbrs_arr[src]; }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> dnbr_arr_t& { return _nbrs_arr[src]; }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& { return _vecs_data; }

    auto get_base_metadata() const -> nlohmann::json {
        nlohmann::json meta;
        meta["graph_type"] = "descent_graph";
        meta["version"] = "1.0";
        meta["num_vertices"] = _num_vertices;
        meta["layer_config"] = {
            {"max_nbr_size", _layer_config.max_nbr_size()},
            {"reserved_nbr_size", _layer_config.reserved_nbr_size()}
        };
        return meta;
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Layer configuration (max_nbr_size and reserved_nbr_size). */
    layer_config_t _layer_config;

    /** @brief Array of neighbors for each vertex. */
    std::vector<dnbr_arr_t> _nbrs_arr;

    /** @brief Const reference to vector data for this layer. */
    const vector_array_t& _vecs_data;

};  // class DescentGraph

}   // namespace cpu
}   // namespace artea
