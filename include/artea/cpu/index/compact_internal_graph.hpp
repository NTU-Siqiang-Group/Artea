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

/*
 * @FilePath: /Artea/include/artea/cpu/index/compact_internal_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Compact internal graph with CSR-format LayerNeighbor storage and inter-layer links.
 */

#pragma once

#include <cstddef>
#include <span>
#include <algorithm>

namespace artea {
namespace cpu {

/**
 * @brief Compact internal graph storing layer neighbors in CSR format
 *        and inter-layer links for hierarchical navigation.
 *
 * Each vertex is assigned a fixed-size neighbor slot of @p max_nbr_size
 * LayerNeighbor entries in the flat @p _csr_nbrs array.
 * The @p _inter_layer_links array maps a vertex's layer_vid to
 * its layer_vid in the next (lower) layer.
 *
 * This class does not provide a direct accessor for the number of valid
 * neighbors per vertex. Instead, it uses a **sentinel-based traversal**
 * pattern: iterate over a vertex's neighbor array from the beginning and
 * stop when encountering a sentinel entry whose @c base_vid equals
 * @c invalid_vertex_id (i.e., @c IndexTraitsT::invalid_lnbr).
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class CompactInternalGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using lnbr_t = typename IndexTraitsT::lnbr_t;

    using csr_lnbrs_t = cache_aligned_container_t<lnbr_t>;
    using inter_layer_links_arr_t = cache_aligned_container_t<vertex_id_t>;

    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct a CompactInternalGraph.
     * @param num_vertices Number of vertices in this layer.
     * @param max_nbr_size Fixed number of neighbor slots per vertex.
     */
    CompactInternalGraph(
        const vertex_num_t num_vertices,
        const vertex_num_t max_nbr_size
    ) :
        _num_vertices(num_vertices),
        _max_nbr_size(max_nbr_size)
    {
        _csr_nbrs.resize(static_cast<size_t>(_num_vertices) * _max_nbr_size);
        std::fill(_csr_nbrs.begin(), _csr_nbrs.end(), IndexTraitsT::invalid_lnbr);

        _inter_layer_links.resize(_num_vertices, invalid_vertex_id);
    }

    // Copying is deleted
    CompactInternalGraph(const CompactInternalGraph&) = delete;
    CompactInternalGraph& operator=(const CompactInternalGraph&) = delete;

    // Default move semantics
    CompactInternalGraph(CompactInternalGraph&&) noexcept = default;
    CompactInternalGraph& operator=(CompactInternalGraph&&) noexcept = default;

    ~CompactInternalGraph() = default;

    // --- Neighbor Access ---

    /**
     * @brief Fetch the neighbor list of a vertex (const).
     * @param src The layer_vid of the source vertex.
     * @return A const span over the fixed-size neighbor slots.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> std::span<const lnbr_t> {
        return std::span<const lnbr_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _max_nbr_size],
            _max_nbr_size
        );
    }

    /**
     * @brief Fetch the neighbor list of a vertex (mutable).
     * @param src The layer_vid of the source vertex.
     * @return A mutable span over the fixed-size neighbor slots.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> std::span<lnbr_t> {
        return std::span<lnbr_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _max_nbr_size],
            _max_nbr_size
        );
    }

    // --- Inter-Layer Links ---

    /**
     * @brief Get the layer_vid of a vertex in the next (lower) layer.
     * @param layer_vid The vertex's layer_vid in the current layer.
     * @return The corresponding layer_vid in the next layer, or invalid_vertex_id if none.
     */
    __attribute__((always_inline))
    auto get_inter_layer_link(const vertex_id_t layer_vid) const -> vertex_id_t {
        return _inter_layer_links[layer_vid];
    }

    /**
     * @brief Set the inter-layer link for a vertex.
     * @param layer_vid The vertex's layer_vid in the current layer.
     * @param next_layer_vid The vertex's layer_vid in the next (lower) layer.
     */
    __attribute__((always_inline))
    auto set_inter_layer_link(const vertex_id_t layer_vid, const vertex_id_t next_layer_vid) -> void {
        _inter_layer_links[layer_vid] = next_layer_vid;
    }

    /**
     * @brief Get the inter-layer links array (const).
     */
    __attribute__((always_inline))
    auto get_inter_layer_links() const -> const inter_layer_links_arr_t& {
        return _inter_layer_links;
    }

    /**
     * @brief Get the inter-layer links array (mutable).
     */
    __attribute__((always_inline))
    auto get_inter_layer_links() -> inter_layer_links_arr_t& {
        return _inter_layer_links;
    }

    // --- Properties ---

    __attribute__((always_inline))
    auto max_nbr_size() const -> vertex_num_t {
        return _max_nbr_size;
    }

    __attribute__((always_inline))
    auto max_nbr_size(const vertex_num_t new_size) -> void {
        _max_nbr_size = new_size;
    }

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() const -> const csr_lnbrs_t& {
        return _csr_nbrs;
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() -> csr_lnbrs_t& {
        return _csr_nbrs;
    }

private:
    /** @brief Number of vertices in this layer. */
    vertex_num_t _num_vertices;

    /** @brief Fixed number of neighbor slots per vertex. */
    vertex_num_t _max_nbr_size;

    /** @brief CSR-format neighbor storage: each vertex occupies _max_nbr_size contiguous slots. */
    csr_lnbrs_t _csr_nbrs;

    /**
     * @brief Inter-layer links: _inter_layer_links[layer_vid] stores
     *        this vertex's layer_vid in the next (lower) layer.
     */
    inter_layer_links_arr_t _inter_layer_links;

};  // class CompactInternalGraph

}   // namespace cpu
}   // namespace artea
