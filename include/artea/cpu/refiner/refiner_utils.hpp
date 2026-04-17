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
 * @FilePath: /Artea/include/artea/cpu/refiner/refiner_utils.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Bridge utilities between dynamic::HierarchicalGraph and
 *               dynamic::RefiningGraph. Hosts the layer ↔ RefiningGraph
 *               conversion helpers (previously methods on
 *               HierarchicalGraph) so the hierarchical graph stays a
 *               pure topology container and all refiner-side operations
 *               live in the refiner namespace.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

#include <artea/common/logger.hpp>
#include <artea/cpu/refiner/propagate_engine.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Static bridge between @c dynamic::HierarchicalGraph layer slots
 *        and @c dynamic::RefiningGraph neighbor arrays.
 *
 * Both helpers iterate the @p refining_graph rows in parallel via
 * @c PropagateEngine::parallel_for_each_vertex, so the RefiningGraph
 * itself holds no iteration logic.
 *
 * @tparam RefinerTraitsT The refiner traits type.
 */
template <typename RefinerTraitsT>
class RefinerUtils {

    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vertex_id_t  = typename RefinerTraitsT::vertex_id_t;
    using layer_id_t   = typename RefinerTraitsT::layer_id_t;
    using nbr_t        = typename RefinerTraitsT::nbr_t;
    using propagate_engine_t = typename RefinerTraitsT::propagate_engine_t;

public:
    /**
     * @brief Populate @p refining_graph's neighbor array from level
     *        @p level_id of @p hier_graph.
     *
     * The caller must already have constructed @p refining_graph with
     * the matching mapping (dense ctor for L0; sparse ctor with the
     * maps from @c hier_graph.collect_layer_vids for upper layers).
     *
     * Per-row writes only; no locks needed because extraction is a
     * phase-boundary operation.
     */
    template <typename HierarchicalGraphT, typename RefiningGraphT>
    static auto fill_refining_graph_from_layer(
        const HierarchicalGraphT& hier_graph,
        RefiningGraphT&           refining_graph,
        const layer_id_t          level_id
    ) -> void {
        // Take min(src_capacity, dest_capacity), so callers can drive the
        // refiner with an RG layer cap that differs from the hier_graph slot cap
        // in either direction. Symmetric with
        // writeback_layer_from_refining_graph.
        const vertex_num_t src_capacity  = hier_graph.max_nbr_size(level_id);
        const vertex_num_t dest_capacity = refining_graph.layer_config().max_nbr_size();
        const vertex_num_t copy_capacity = std::min(src_capacity, dest_capacity);

        propagate_engine_t::parallel_for_each_vertex(
            refining_graph,
            [&](const vertex_id_t /*layer_vid*/, const vertex_id_t storage_vid) {
                // Skip vids that were never assign_layer'd. Possible when
                // the caller constructed a dense RG with a vector array
                // larger than the assigned set (e.g. reusing the global
                // dataset for a partial fixture). Not produced by the
                // standard add_vertices → assign_layer flow, so warn.
                if (!hier_graph.is_vertex_assigned(storage_vid)) {
                    ARTEA_WARN(fmt::format(
                        "fill_refining_graph_from_layer: skipping "
                        "unassigned vid={} (level_id={})",
                        storage_vid, level_id));
                    return;
                }
                const auto src = hier_graph.fetch_layer_nbrs(storage_vid, level_id);
                auto& dst = refining_graph.fetch_nbrs(storage_vid);
                dst.clear();
                for (vertex_num_t i = 0; i < copy_capacity && i < src.size(); ++i) {
                    if (src[i].is_invalid()) break;
                    dst.push_back(src[i]);
                }
            });
    }

    /**
     * @brief Write @p refining_graph's neighbor lists back into level
     *        @p level_id of @p hier_graph.
     *
     * Per-vertex spinlock guards each write via
     * @c hier_graph.with_locked_nbrs; in practice writeback runs as a
     * phase boundary so contention is nil. The destination slot is
     * truncated to its capacity for @p level_id (handles the L0 2× cap
     * automatically) and terminated with the invalid sentinel.
     */
    template <typename HierarchicalGraphT, typename RefiningGraphT>
    static auto writeback_layer_from_refining_graph(
        HierarchicalGraphT& hier_graph,
        RefiningGraphT&     refining_graph,
        const layer_id_t    level_id
    ) -> void {
        propagate_engine_t::parallel_for_each_vertex(
            refining_graph,
            [&](const vertex_id_t /*layer_vid*/, const vertex_id_t storage_vid) {
                if (!hier_graph.is_vertex_assigned(storage_vid)) {
                    ARTEA_WARN(fmt::format(
                        "writeback_layer_from_refining_graph: skipping "
                        "unassigned vid={} (level_id={})",
                        storage_vid, level_id));
                    return;
                }
                const auto& src = refining_graph.fetch_nbrs(storage_vid);
                hier_graph.with_locked_nbrs(storage_vid, level_id,
                    [&](std::span<nbr_t> dst, vertex_num_t /*old_cnt*/) {
                        const std::size_t copy_n =
                            std::min(src.size(), dst.size());
                        for (std::size_t i = 0; i < copy_n; ++i) {
                            dst[i] = src[i];
                        }
                        if (copy_n < dst.size()) {
                            dst[copy_n] = nbr_t::make_invalid_nbr();
                        }
                    });
            });
    }

};  // class RefinerUtils

}   // namespace cpu
}   // namespace artea
