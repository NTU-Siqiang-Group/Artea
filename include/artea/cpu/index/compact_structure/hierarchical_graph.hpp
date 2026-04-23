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
 * @FilePath: /Artea/include/artea/cpu/index/compact_structure/hierarchical_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Read-only topology-only snapshot of a
 *               dynamic::HierarchicalGraph. Mirrors the dynamic storage
 *               layout byte-for-byte except that:
 *                 - neighbor entries are raw vertex_id_t (no distance),
 *                 - per-vertex spinlock is removed,
 *                 - arenas are sized exactly and never grow.
 *               Slot offsets are preserved from the source, so the
 *               compactor can copy VertexInfo verbatim without
 *               recomputing layout.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {
namespace compact {

/**
 * @brief Read-only hierarchical graph storing topology only.
 *
 * Arenas are per-@c highest_level_id @c cache_aligned_container_t buffers
 * sized once at construction. Each vertex's @c VertexInfo holds its
 * @c highest_level_id plus a @c slot_offset (in @c vertex_id_t units)
 * into its group's arena, preserved verbatim from the dynamic source.
 *
 * Level layout inside a slot (high → low, ul vs bl independent):
 *
 *   [ level H vids   | level H-1 vids  | ... | level 1 vids   | level 0 vids  ]
 *     ul_max_nbr_size  ul_max_nbr_size         ul_max_nbr_size  bl_max_nbr_size
 *
 * Neighbor arrays are sentinel-terminated with @c invalid_vertex_id.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraph {

public:
    // Public so router-side adapters can read these typedefs without
    // re-deriving them from IndexTraits.
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t  = typename IndexTraitsT::vertex_id_t;
    using layer_num_t  = typename IndexTraitsT::layer_num_t;
    using layer_id_t   = typename IndexTraitsT::layer_id_t;

private:
    using vid_arena_container_t = cache_aligned_container_t<vertex_id_t>;

public:
    /** @brief Marks this graph as a compact (read-only, post-compaction)
     *         storage. Routers branch on this in @c detail::make_layer_range
     *         to pick the right NeighborRange adapter. */
    static constexpr bool is_compacted = true;

    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;

    /** @brief Sentinel for vertices that have not been assigned to any
     *         level. Matches the dynamic graph's convention so the
     *         compactor can copy @c highest_level_id verbatim. */
    static constexpr layer_id_t unassigned_highest_level_id =
        std::numeric_limits<layer_id_t>::max();

    /**
     * @brief Per-vertex row. Strict subset of
     *        @c dynamic::HierarchicalGraph::VertexInfo (drops the spinlock);
     *        the semantics and units of @c slot_offset exactly match
     *        the dynamic version.
     */
    struct VertexInfo {
        layer_id_t  highest_level_id = unassigned_highest_level_id;
        std::size_t slot_offset      = 0;   // in vertex_id_t units
    };

    /**
     * @brief Construct a compact graph sized to hold every vid's slot
     *        from a source dynamic graph.
     *
     * @param max_restrict_level          Inclusive upper bound of
     *                                      per-vertex @c highest_level_id.
     * @param ul_max_nbr_size               Per-vertex capacity at upper
     *                                      levels (level_id > 0).
     * @param bl_max_nbr_size               Per-vertex capacity at the
     *                                      bottom level (L0). Independent
     *                                      of @p ul_max_nbr_size.
     * @param num_vertices                  Total vertex count
     *                                      (sizes @c _vertex_info_table).
     * @param arena_vid_capacity_per_group  Per-arena size in
     *                                      @c vertex_id_t elements. Must
     *                                      be >= the maximum @c slot_offset
     *                                      that the compactor will write
     *                                      into that arena.
     */
    HierarchicalGraph(
        const layer_num_t              max_restrict_level,
        const vertex_num_t             ul_max_nbr_size,
        const vertex_num_t             bl_max_nbr_size,
        const vertex_num_t             num_vertices,
        std::vector<std::size_t>       arena_vid_capacity_per_group
    ) :
        _max_restrict_level(max_restrict_level),
        _ul_max_nbr_size(ul_max_nbr_size),
        _bl_max_nbr_size(bl_max_nbr_size),
        _num_vertices(num_vertices),
        _entry_point_vid(invalid_vertex_id),
        _arenas(static_cast<std::size_t>(max_restrict_level) + 1),
        _vids_by_highest_level(static_cast<std::size_t>(max_restrict_level) + 1),
        _vertex_info_table(num_vertices)
    {
        for (std::size_t h = 0; h < _arenas.size(); ++h) {
            _arenas[h].resize(arena_vid_capacity_per_group[h]);
            std::fill(_arenas[h].begin(), _arenas[h].end(),
                      invalid_vertex_id);
        }
    }

    HierarchicalGraph(const HierarchicalGraph&)            = delete;
    HierarchicalGraph& operator=(const HierarchicalGraph&) = delete;
    HierarchicalGraph(HierarchicalGraph&&)                 = default;
    HierarchicalGraph& operator=(HierarchicalGraph&&)      = default;

    ~HierarchicalGraph() = default;

    // =================================================================
    //   Read-only query API (mirrors dynamic::HierarchicalGraph)
    // =================================================================

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto max_restrict_level() const -> layer_id_t {
        return _max_restrict_level;
    }

    /** @brief Per-vertex capacity at every upper layer. */
    __attribute__((always_inline))
    auto ul_max_nbr_size() const -> vertex_num_t {
        return _ul_max_nbr_size;
    }

    /** @brief Per-vertex capacity at the bottom layer (L0). */
    __attribute__((always_inline))
    auto bl_max_nbr_size() const -> vertex_num_t {
        return _bl_max_nbr_size;
    }

    /** @brief Per-vertex capacity at @p l (bl for L0, ul elsewhere). */
    __attribute__((always_inline))
    auto max_nbr_size(const layer_id_t l) const -> vertex_num_t {
        return (l == 0) ? _bl_max_nbr_size : _ul_max_nbr_size;
    }

    __attribute__((always_inline))
    auto get_highest_level_id(const vertex_id_t vid) const -> layer_id_t {
        return _vertex_info_table[vid].highest_level_id;
    }

    /**
     * @brief Return a const span over the neighbor array of @p vid at
     *        level @p level_id. Offset and length are computed
     *        branch-free, matching the dynamic graph's formula.
     *
     * Precondition: @c level_id <= get_highest_level_id(vid).
     */
    __attribute__((always_inline))
    auto fetch_layer_nbrs(
        const vertex_id_t vid,
        const layer_id_t  level_id
    ) const -> std::span<const vertex_id_t> {
        const auto& vinfo = _vertex_info_table[vid];
        const layer_id_t H = vinfo.highest_level_id;
        const vertex_id_t* slot_base = _arenas[H].data() + vinfo.slot_offset;

        const std::size_t level_offset =
            static_cast<std::size_t>(H - level_id) * _ul_max_nbr_size;
        const std::size_t level_nbrs_count =
            (level_id == 0)
                ? static_cast<std::size_t>(_bl_max_nbr_size)
                : static_cast<std::size_t>(_ul_max_nbr_size);
        return std::span<const vertex_id_t>(slot_base + level_offset, level_nbrs_count);
    }

    /**
     * @brief Scan forward until the first @c invalid_vertex_id sentinel
     *        in the level-@p level_id slot of @p vid.
     *        O(max_nbr_size(level_id)).
     */
    __attribute__((always_inline))
    auto num_valid_nbrs(
        const vertex_id_t vid,
        const layer_id_t  level_id
    ) const -> vertex_num_t {
        const auto nbrs = fetch_layer_nbrs(vid, level_id);
        vertex_num_t count = 0;
        for (const vertex_id_t nvid : nbrs) {
            if (nvid == invalid_vertex_id) break;
            ++count;
        }
        return count;
    }

    /**
     * @brief Vids whose @c highest_level_id == @p h.
     */
    __attribute__((always_inline))
    auto get_vids_with_highest_level(const layer_id_t h) const -> std::span<const vertex_id_t> 
    {
        return std::span<const vertex_id_t>(_vids_by_highest_level[h].data(), _vids_by_highest_level[h].size());
    }

    /**
     * @brief Largest @c h with a non-empty bucket, or
     *        @c unassigned_highest_level_id if every bucket is empty.
     *
     * The compactor (@c HierarchicalGraphCompactor) trims the compact
     * graph's @c _max_restrict_level down to exactly the source's
     * top_occupied_level_id at construction time, so the invariant
     * here is: any non-empty compact graph has
     * @c top_occupied_level_id == _max_restrict_level. We therefore
     * return that constant directly instead of scanning buckets.
     */
    auto top_occupied_level_id() const -> layer_id_t {
        return (_num_vertices == 0)
            ? unassigned_highest_level_id
            : _max_restrict_level;
    }

    /**
     * @brief Pre-computed hierarchical entry-point vid. Set by
     *        @c HierarchicalGraphCompactor to the top-bucket vid
     *        closest to the centroid of the top bucket; consumed by
     *        @c HierarchicalGraphRouter in place of the sampling-based
     *        seeding (@c sample_entries / @c sample_single_entry).
     *        Returns @c invalid_vertex_id when unset (empty graph).
     */
    __attribute__((always_inline))
    auto entry_point_vid() const -> vertex_id_t {
        return _entry_point_vid;
    }

    // =================================================================
    //   Compactor-facing mutators
    // =================================================================

    /** @brief Set the cached entry-point vid. Called by the compactor
     *         after it has materialized the top bucket. */
    __attribute__((always_inline))
    auto set_entry_point_vid(const vertex_id_t vid) -> void {
        _entry_point_vid = vid;
    }

    /**
     * @brief Mutable access to the vertex-info table. Used by
     *        HierarchicalGraphCompactor to stream in per-vertex
     *        (highest_level_id, slot_offset) copied from the source.
     */
    __attribute__((always_inline))
    auto get_vertex_info_table_mut() -> std::vector<VertexInfo>& {
        return _vertex_info_table;
    }

    /**
     * @brief Mutable access to the per-group vid buckets. Used by
     *        HierarchicalGraphCompactor to populate the groups by
     *        bulk-copying the dynamic source's concurrent buckets.
     */
    __attribute__((always_inline))
    auto get_vids_by_highest_level_mut()
        -> std::vector<std::vector<vertex_id_t>>&
    {
        return _vids_by_highest_level;
    }

    /** @brief Raw base pointer of arena @p h (for compactor writes). */
    __attribute__((always_inline))
    auto arena_base(const layer_id_t h) -> vertex_id_t* {
        return _arenas[h].data();
    }

private:
    layer_num_t  _max_restrict_level;
    vertex_num_t _ul_max_nbr_size;
    vertex_num_t _bl_max_nbr_size;
    vertex_num_t _num_vertices;

    /** @brief Cached hierarchical entry point — the vid closest to the
     *         top-bucket centroid. Populated by the compactor; consumed
     *         by the router in place of random sampling. */
    vertex_id_t  _entry_point_vid;

    /** @brief One arena per highest_level_id in
     *         @c [0, _max_restrict_level]. */
    std::vector<vid_arena_container_t>    _arenas;

    /** @brief _vids_by_highest_level[h] lists every vid with
     *         highest_level_id == h. Copied from the source at compact
     *         time; not mutated thereafter. */
    std::vector<std::vector<vertex_id_t>> _vids_by_highest_level;

    /** @brief Per-vertex row, indexed by vid. */
    std::vector<VertexInfo>               _vertex_info_table;

};  // class HierarchicalGraph

}   // namespace compact
}   // namespace cpu
}   // namespace artea
