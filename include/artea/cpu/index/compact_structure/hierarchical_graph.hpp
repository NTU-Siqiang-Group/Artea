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

// Compact topology: groups contain vertices with the same final highest level.
// Each group stores N_h * (h + 1) CSR rows, ordered highest level down to L0.
// Global 64-bit offsets address one array of valid 32-bit neighbor IDs, without sentinel padding.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <artea/cpu/containers/allocator.hpp>

namespace artea::cpu::compact {

template <typename IndexTraitsT>
class HierarchicalGraph {
public:
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using nbr_offset_t = std::uint64_t;

    static constexpr bool is_compacted = true;
    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;
    static constexpr layer_id_t invalid_level_id = std::numeric_limits<layer_id_t>::max();

    struct VertexInfo {
        layer_id_t highest_level = invalid_level_id;
        vertex_id_t local_vid = invalid_vertex_id;
    };

    // Group counts describe final buckets, including vertices demoted by top-layer trimming.
    HierarchicalGraph(layer_id_t top_level_id, vertex_num_t ul_max_nbr_size,
                      vertex_num_t bl_max_nbr_size, vertex_num_t num_vertices,
                      const std::vector<std::size_t>& vertex_counts_per_group)
        : _top_level_id(top_level_id), _ul_max_nbr_size(ul_max_nbr_size),
          _bl_max_nbr_size(bl_max_nbr_size), _num_vertices(num_vertices),
          _nbr_offsets(vertex_counts_per_group.size()),
          _vids_by_highest_level(vertex_counts_per_group.size()), _vertex_info_table(num_vertices) {
        if (vertex_counts_per_group.size() != static_cast<std::size_t>(top_level_id) + 1) {
            throw std::invalid_argument("HierarchicalGraph: group count does not match top level");
        }
        std::size_t assigned_vertices = 0;
        for (std::size_t highest_level = 0; highest_level < _nbr_offsets.size(); ++highest_level) {
            const std::size_t vertex_count = vertex_counts_per_group[highest_level];
            const std::size_t levels_per_vertex = highest_level + 1;
            auto& offsets = _nbr_offsets[highest_level];
            if (vertex_count > num_vertices - assigned_vertices || vertex_count >= invalid_vertex_id ||
                vertex_count > (offsets.max_size() - 1) / levels_per_vertex) {
                throw std::length_error("HierarchicalGraph: CSR row count exceeds storage limit");
            }
            assigned_vertices += vertex_count;
            offsets.resize(vertex_count * levels_per_vertex + 1);
            // The aligned allocator deliberately skips initialization of trivial elements.
            std::fill(offsets.begin(), offsets.end(), nbr_offset_t{0});
        }
    }

    HierarchicalGraph(const HierarchicalGraph&) = delete;
    HierarchicalGraph& operator=(const HierarchicalGraph&) = delete;
    HierarchicalGraph(HierarchicalGraph&&) = default;
    HierarchicalGraph& operator=(HierarchicalGraph&&) = default;

    auto get_num_vertices() const -> vertex_num_t { return _num_vertices; }
    auto ul_max_nbr_size() const -> vertex_num_t { return _ul_max_nbr_size; }
    auto bl_max_nbr_size() const -> vertex_num_t { return _bl_max_nbr_size; }
    auto max_nbr_size(layer_id_t level_id) const -> vertex_num_t {
        return level_id == 0 ? _bl_max_nbr_size : _ul_max_nbr_size;
    }
    auto get_highest_level_id(vertex_id_t vid) const -> layer_id_t {
        return _vertex_info_table[vid].highest_level;
    }
    auto get_vertex_info(vertex_id_t vid) const -> const VertexInfo& { return _vertex_info_table[vid]; }
    auto get_nbr_offsets(layer_id_t highest_level) const -> std::span<const nbr_offset_t> {
        return _nbr_offsets[highest_level];
    }
    auto neighbor_ids() const -> std::span<const vertex_id_t> { return _nbrs_arr; }

    // Precondition: vid is assigned, and level_id <= its highest_level. Lookup takes O(1).
    __attribute__((always_inline))
    auto fetch_level_nbrs(vertex_id_t vid, layer_id_t level_id) const -> std::span<const vertex_id_t> {
        const auto& info = _vertex_info_table[vid];
        const std::size_t row_index = static_cast<std::size_t>(info.local_vid) *
                                     (static_cast<std::size_t>(info.highest_level) + 1) +
                                     (info.highest_level - level_id);
        const auto& offsets = _nbr_offsets[info.highest_level];
        return std::span<const vertex_id_t>(_nbrs_arr).subspan(
            offsets[row_index], offsets[row_index + 1] - offsets[row_index]);
    }
    auto num_valid_nbrs(vertex_id_t vid, layer_id_t level_id) const -> vertex_num_t {
        return static_cast<vertex_num_t>(fetch_level_nbrs(vid, level_id).size());
    }
    auto get_vids_with_highest_level(layer_id_t highest_level) const -> std::span<const vertex_id_t> {
        return _vids_by_highest_level[highest_level];
    }
    auto get_top_level_vids() const -> std::span<const vertex_id_t> {
        return _num_vertices == 0 ? std::span<const vertex_id_t>{} : _vids_by_highest_level[_top_level_id];
    }
    auto top_occupied_level_id() const -> layer_id_t {
        return _num_vertices == 0 ? invalid_level_id : _top_level_id;
    }
    auto entry_point_vid() const -> vertex_id_t { return _entry_point_vid; }

    // Actual vector allocations, excluding allocator bookkeeping and the graph object itself.
    auto allocated_storage_bytes() const -> std::size_t {
        std::size_t bytes = _vertex_info_table.capacity() * sizeof(VertexInfo) +
                            _nbrs_arr.capacity() * sizeof(vertex_id_t) +
                            _nbr_offsets.capacity() * sizeof(offset_container_t) +
                            _vids_by_highest_level.capacity() * sizeof(std::vector<vertex_id_t>);
        for (const auto& offsets : _nbr_offsets) bytes += offsets.capacity() * sizeof(nbr_offset_t);
        for (const auto& bucket : _vids_by_highest_level) bytes += bucket.capacity() * sizeof(vertex_id_t);
        return bytes;
    }

    // Builder-only APIs: write row lengths first, finalize offsets once, then fill disjoint neighbor spans.
    auto set_entry_point_vid(vertex_id_t vid) -> void { _entry_point_vid = vid; }
    auto get_vertex_info_table_mut() -> std::vector<VertexInfo>& { return _vertex_info_table; }
    auto get_vids_by_highest_level_mut() -> std::vector<std::vector<vertex_id_t>>& {
        return _vids_by_highest_level;
    }
    auto get_nbr_offsets_mut(layer_id_t highest_level) -> std::span<nbr_offset_t> {
        return _nbr_offsets[highest_level];
    }
    auto allocate_neighbors_from_row_counts() -> void {
        nbr_offset_t total_neighbors = 0;
        const auto max_neighbors = _nbrs_arr.max_size();
        for (auto& offsets : _nbr_offsets) {
            for (std::size_t row_index = 0; row_index + 1 < offsets.size(); ++row_index) {
                const nbr_offset_t row_length = offsets[row_index];
                if (row_length > max_neighbors - total_neighbors) {
                    throw std::length_error("HierarchicalGraph: neighbor array exceeds storage limit");
                }
                offsets[row_index] = total_neighbors;
                total_neighbors += row_length;
            }
            offsets.back() = total_neighbors;
        }
        _nbrs_arr.resize(static_cast<std::size_t>(total_neighbors));
    }
    auto fetch_level_nbrs_mut(vertex_id_t vid, layer_id_t level_id) -> std::span<vertex_id_t> {
        const auto neighbors = fetch_level_nbrs(vid, level_id);
        return {const_cast<vertex_id_t*>(neighbors.data()), neighbors.size()};
    }

private:
    /** @brief Cache-aligned storage for 64-bit CSR offsets. */
    using offset_container_t = cache_aligned_container_t<nbr_offset_t>;

    /** @brief Highest retained level; L0 serves as the empty-graph placeholder. */
    layer_id_t _top_level_id;

    /** @brief Configured neighbor-count limit for each upper-level row; CSR stores only valid neighbors. */
    vertex_num_t _ul_max_nbr_size;

    /** @brief Configured neighbor-count limit for each L0 row; CSR stores only valid neighbors. */
    vertex_num_t _bl_max_nbr_size;

    /** @brief Number of global vertex IDs, including unassigned vertices; sizes the vertex-info table. */
    vertex_num_t _num_vertices;

    /** @brief Cached top-bucket vertex closest to its centroid; invalid_vertex_id when unset. */
    vertex_id_t _entry_point_vid = invalid_vertex_id;

    /**
     * @brief Per-highest-level CSR offsets into @c _nbrs_arr, measured in neighbor-ID elements.
     * Group h contains N_h * (h + 1) + 1 offsets, with each local vertex's rows ordered h down to L0.
     * Row index = local_vid * (h + 1) + (h - level_id); consecutive offsets delimit its neighbor span.
     * Empty groups retain one end offset. Before finalization, row-start entries hold neighbor counts.
     */
    std::vector<offset_container_t> _nbr_offsets;

    /** @brief Global array of valid neighbor IDs in CSR row order, preserving each row's neighbor order. */
    cache_aligned_container_t<vertex_id_t> _nbrs_arr;

    /** @brief Bucket h lists vertices whose final highest level is h; each vertex's local_vid indexes it. */
    std::vector<std::vector<vertex_id_t>> _vids_by_highest_level;

    /**
     * @brief Metadata indexed by global vid: highest_level and local_vid; unassigned entries use sentinels.
     */
    std::vector<VertexInfo> _vertex_info_table;
};

}  // namespace artea::cpu::compact
