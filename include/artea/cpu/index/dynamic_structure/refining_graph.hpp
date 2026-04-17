/*
 * @FilePath: /Artea/include/artea/cpu/index/refining_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: CRTP base descent graph structure for graph-based index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <nlohmann/json.hpp>


namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Descent Graph: ordinary graph type that stores graph topology and
 *        layer config. Used by composition (conv_graph::IndexStructure,
 *        knn_graph::IndexStructure hold one through unique_ptr).
 *
 *        Optionally carries a sparse vid mapping (local_to_global /
 *        global_to_local) so a single layer of HierarchicalGraph can be
 *        extracted into a standalone RefiningGraph for refinement.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class RefiningGraph {

protected:
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;

public:
    /**
     * @brief Construct a new Descent Graph object.
     * @param vecs_data Reference to the vector data for this layer.
     * @param layer_config Layer configuration (max_nbr_size and reserved_nbr_size).
     */
    /**
     * @brief Dense (identity-mapped) constructor. Every global vid in
     *        @p vecs_data is a row in @c _nbrs_arr; @c fetch_nbrs(vid)
     *        directly indexes by the global vid.
     */
    RefiningGraph(
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

    /**
     * @brief Sparse constructor for a subset of @p global_vecs. Only the
     *        vids in @p local_to_global have rows in @c _nbrs_arr (one
     *        row per local index); @c fetch_nbrs(storage_vid) translates
     *        through @c _global_to_local on the hot path.
     *
     * @param global_vecs     Borrowed reference to the global vector array.
     *                        Neighbor entries continue to carry global vids,
     *                        so vec_data lookups via @c nbr.get_vid() stay
     *                        correct without further translation.
     * @param layer_config    Per-vertex neighbor capacity for this layer.
     * @param local_to_global Local row index → global vid (size = N_local).
     *                        Order defines @c _nbrs_arr layout.
     * @param global_to_local Global vid → local row index (size = N_global);
     *                        entries for vids not in this layer must be
     *                        @c invalid_vertex_id.
     */
    RefiningGraph(
        const vector_array_t&         global_vecs,
        const layer_config_t          layer_config,
        std::vector<vertex_id_t>      local_to_global,
        std::vector<vertex_id_t>      global_to_local
    ) :
        _num_vertices(static_cast<vertex_num_t>(local_to_global.size())),
        _layer_config(layer_config),
        _vecs_data(global_vecs),
        _local_to_global(std::move(local_to_global)),
        _global_to_local(std::move(global_to_local))
    {
        _nbrs_arr.resize(_num_vertices);
        for (vertex_num_t i = 0; i < _num_vertices; ++i) {
            _nbrs_arr[i].reserve(layer_config.reserved_nbr_size());
        }
    }

    RefiningGraph(const RefiningGraph&) = delete;
    RefiningGraph& operator=(const RefiningGraph&) = delete;

    RefiningGraph(RefiningGraph&& other) noexcept
        : _num_vertices(other._num_vertices),
          _layer_config(other._layer_config),
          _nbrs_arr(std::move(other._nbrs_arr)),
          _vecs_data(other._vecs_data),
          _local_to_global(std::move(other._local_to_global)),
          _global_to_local(std::move(other._global_to_local))
    {}

    RefiningGraph& operator=(RefiningGraph&& other) noexcept {
        _num_vertices = other._num_vertices;
        _layer_config = other._layer_config;
        _nbrs_arr = std::move(other._nbrs_arr);
        _local_to_global = std::move(other._local_to_global);
        _global_to_local = std::move(other._global_to_local);
        // _vecs_data is a reference, cannot be reseated
        return *this;
    }

    // --- Public Interface ---

    /** @brief Number of rows in @c _nbrs_arr. Equals the global vertex
     *         count in identity mode and the participating-vid count
     *         in sparse mode. */
    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t { return _num_vertices; }

    /** @brief True iff every global vid maps directly to its own row in
     *         @c _nbrs_arr (no translation needed). */
    __attribute__((always_inline))
    auto is_identity_mapped() const -> bool {
        return _global_to_local.empty();
    }

    /** @brief Local row index → global vid. Empty in identity mode. */
    __attribute__((always_inline))
    auto local_to_global() const -> const std::vector<vertex_id_t>& {
        return _local_to_global;
    }

    /** @brief Global vid → local row index. Empty in identity mode;
     *         entries for non-participating vids are @c invalid_vertex_id. */
    __attribute__((always_inline))
    auto global_to_local() const -> const std::vector<vertex_id_t>& {
        return _global_to_local;
    }

    /** @brief Translate a single global vid to its local row index
     *         (identity passthrough in dense mode). */
    __attribute__((always_inline))
    auto local_id_of(const vertex_id_t storage_vid) const -> vertex_id_t {
        return _global_to_local.empty() ? storage_vid : _global_to_local[storage_vid];
    }

    /** @brief Translate a local row index to its global vid (identity
     *         passthrough in dense mode). */
    __attribute__((always_inline))
    auto get_storage_vid(const vertex_num_t local_idx) const -> vertex_id_t {
        return _local_to_global.empty()
            ? static_cast<vertex_id_t>(local_idx)
            : _local_to_global[local_idx];
    }

    __attribute__((always_inline))
    auto layer_config() const -> const layer_config_t& { return _layer_config; }

    __attribute__((always_inline))
    auto layer_config() -> layer_config_t& { return _layer_config; }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<nbr_arr_t>& { return _nbrs_arr; }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<nbr_arr_t>& { return _nbrs_arr; }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t storage_vid) const -> const nbr_arr_t& {
        return _global_to_local.empty()
            ? _nbrs_arr[storage_vid]
            : _nbrs_arr[_global_to_local[storage_vid]];
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t storage_vid) -> nbr_arr_t& {
        return _global_to_local.empty()
            ? _nbrs_arr[storage_vid]
            : _nbrs_arr[_global_to_local[storage_vid]];
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& { return _vecs_data; }

    auto get_base_metadata() const -> nlohmann::json {
        nlohmann::json meta;
        meta["graph_type"] = "refining_graph";
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
    std::vector<nbr_arr_t> _nbrs_arr;

    /** @brief Const reference to vector data for this layer. */
    const vector_array_t& _vecs_data;

    /** @brief Local row index → global vid. Empty in identity mode. */
    std::vector<vertex_id_t> _local_to_global;

    /** @brief Global vid → local row index. Empty in identity mode;
     *         entries for non-participating vids are @c invalid_vertex_id. */
    std::vector<vertex_id_t> _global_to_local;

};  // class RefiningGraph

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
