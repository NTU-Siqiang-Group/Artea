/*
 * @FilePath: /Artea/include/artea/cpu/index/flat_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-01-31
 * @Description: Flat graph structure for graph-based index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Flat graph structure for graph-based index.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class FlatGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::conv_graph::propagate_config_t;
    using pruning_config_t = typename IndexTraitsT::conv_graph::pruning_config_t;

public:
    /**
     * @brief Construct a new Flat Graph object.
     * @param vecs_data Reference to the vector data for this layer.
     * @param layer_config Layer configuration (max_nbr_size and reserved_nbr_size).
     * @param pruning_config Pruning configuration (scale_coeffs and shifted_coeffs).
     * @param propagate_config Propagation configuration (num_build_loops, num_triangle_updater_iters, prefill_ratio).
     */
    FlatGraph(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) :
        _num_vertices(vecs_data.get_num_vecs()),
        _layer_config(layer_config),
        _pruning_config(pruning_config),
        _propagate_config(propagate_config),
        _vecs_data(vecs_data)
    {
        _nbrs_arr.resize(_num_vertices);
        for (vertex_num_t i = 0; i < _num_vertices; ++i) {
            _nbrs_arr[i].reserve(layer_config.reserved_nbr_size());
        }
    }

    // Copying is deleted
    FlatGraph(const FlatGraph&) = delete;
    FlatGraph& operator=(const FlatGraph&) = delete;

    // default move constructor and assignment
    FlatGraph(FlatGraph&&) noexcept = default;
    FlatGraph& operator=(FlatGraph&&) noexcept = default;

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto layer_config() const -> const layer_config_t& {
        return _layer_config;
    }

    __attribute__((always_inline))
    auto layer_config() -> layer_config_t& {
        return _layer_config;
    }

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& {
        return _pruning_config;
    }

    __attribute__((always_inline))
    auto pruning_config() -> pruning_config_t& {
        return _pruning_config;
    }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& {
        return _propagate_config;
    }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& {
        return _propagate_config;
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<nbr_arr_t>& {
        return _nbrs_arr;
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<nbr_arr_t>& {
        return _nbrs_arr;
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> const nbr_arr_t& {
        return _nbrs_arr[src];
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> nbr_arr_t& {
        return _nbrs_arr[src];
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _vecs_data;
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Layer configuration (max_nbr_size and reserved_nbr_size). */
    layer_config_t _layer_config;

    /** @brief Pruning configuration. */
    pruning_config_t _pruning_config;

    /** @brief Propagation configuration. */
    propagate_config_t _propagate_config;

    /** @brief Array of neighbors for each vertex. */
    std::vector<nbr_arr_t> _nbrs_arr;

    /** @brief Const reference to vector data for this layer. */
    const vector_array_t& _vecs_data;

};  // class FlatGraph

}   // namespace cpu
}   // namespace artea