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
 * @FilePath: /Artea/include/artea/cpu/index/compactor/hierarchical_graph_compactor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Serial compactor: dynamic::HierarchicalGraph →
 *               compact::HierarchicalGraph. Preserves per-vertex
 *               slot_offset verbatim so the compact layout is
 *               byte-compatible with the source (only swapping nbr_t
 *               entries for vertex_id_t and dropping the spinlock).
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Convert a dynamic hierarchical graph into its topology-only
 *        compact counterpart.
 *
 * The compactor trusts the dynamic graph's slot layout. For each vid:
 *   - Copy @c (highest_level_id, slot_offset) into compact's
 *     @c VertexInfo.
 *   - For every level @c l in @c [0, highest_level_id], scan the
 *     dynamic slot until the first invalid sentinel and copy
 *     @c nbr.get_vid() into the matching compact slot position.
 *
 * Per-vid work is independent across vids, so the straight serial loop
 * is trivially parallelizable (TODO: TBB parallel_for once correctness
 * is confirmed).
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraphCompactor {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t  = typename IndexTraitsT::vertex_id_t;
    using layer_num_t  = typename IndexTraitsT::layer_num_t;
    using layer_id_t   = typename IndexTraitsT::layer_id_t;

    using dynamic      = typename IndexTraitsT::dynamic;
    using compact      = typename IndexTraitsT::compact;

public:
    /**
     * @brief Serially compact @p src into a fresh compact graph.
     *
     * @param src  Dynamic source graph. Must have @c assign_layer
     *             completed for every vid in @c [0, src.get_num_vertices()).
     * @return A new compact::HierarchicalGraph with identical topology.
     */
    static auto compact_graph(
        const typename dynamic::hierarchical_graph_t& src
    ) -> typename compact::hierarchical_graph_t {
        const layer_id_t   max_h        = src.max_highest_level_id();
        const vertex_num_t num_vertices = src.get_num_vertices();
        const vertex_num_t max_nbr_size = src.max_nbr_size();

        // 1. Size each compact arena exactly to match the source's
        //    current bump-allocator capacity. slot_capacity >= num
        //    claimed slots, so slot_offsets copied verbatim from src
        //    always fit.
        std::vector<std::size_t> arena_vid_capacity(
            static_cast<std::size_t>(max_h) + 1);
        for (layer_id_t h = 0; h <= max_h; ++h) {
            const std::size_t slot_nbrs_count =
                static_cast<std::size_t>(h + 2) *
                static_cast<std::size_t>(max_nbr_size);
            arena_vid_capacity[h] =
                static_cast<std::size_t>(src.get_arena_slot_capacity(h)) *
                slot_nbrs_count;
        }

        typename compact::hierarchical_graph_t result(
            max_h, max_nbr_size, num_vertices,
            std::move(arena_vid_capacity));

        // 2. Copy per-vertex (highest_level_id, slot_offset) verbatim.
        auto& compact_vit = result.get_vertex_info_table_mut();
        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            compact_vit[vid].highest_level_id = src.get_highest_level_id(vid);
            compact_vit[vid].slot_offset      = src.get_slot_offset(vid);
        }

        // 3. Copy per-group vid buckets (tbb::concurrent_vector →
        //    std::vector).
        auto& compact_buckets = result.get_vids_by_highest_level_mut();
        for (layer_id_t h = 0; h <= max_h; ++h) {
            const auto& src_bucket = src.get_vids_with_highest_level(h);
            compact_buckets[h].reserve(src_bucket.size());
            for (const vertex_id_t vid : src_bucket) {
                compact_buckets[h].push_back(vid);
            }
        }

        // 4. Transcribe per-level neighbor slots: nbr_t → vertex_id_t.
        //    Walking each group bucket keeps the access pattern arena-
        //    friendly.
        for (layer_id_t h = 0; h <= max_h; ++h) {
            for (const vertex_id_t vid : compact_buckets[h]) {
                const std::size_t slot_offset =
                    compact_vit[vid].slot_offset;
                vertex_id_t* slot_base = result.arena_base(h) + slot_offset;

                for (layer_id_t cur_level = 0;
                     cur_level <= h;
                     ++cur_level)
                {
                    const auto src_layer_nbrs =
                        src.fetch_layer_nbrs(vid, cur_level);
                    const vertex_num_t count =
                        src.num_valid_nbrs(vid, cur_level);

                    const std::size_t level_offset =
                        static_cast<std::size_t>(h - cur_level) * max_nbr_size;
                    vertex_id_t* dst = slot_base + level_offset;

                    for (vertex_num_t i = 0; i < count; ++i) {
                        dst[i] = src_layer_nbrs[i].get_vid();
                    }
                    // Trailing sentinels were pre-filled by the compact
                    // graph's constructor.
                }
            }
        }

        return result;
    }

};  // class HierarchicalGraphCompactor

}   // namespace cpu
}   // namespace artea
