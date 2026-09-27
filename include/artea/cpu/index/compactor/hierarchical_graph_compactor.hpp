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
 * @Description: Parallel compactor: dynamic::HierarchicalGraph →
 *               compact::HierarchicalGraph.
 *
 *               Beyond the basic topology copy, the compactor also
 *                 (a) trims sparse top layers — any top bucket whose
 *                     population is below @p min_layer_cap is removed
 *                     and its apex vids are demoted to the new top,
 *                 (b) precomputes a hierarchical entry point as the
 *                     top-bucket vid closest to its centroid, so the
 *                     router can skip runtime sampling on every query.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <limits>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Convert a dynamic hierarchical graph into its topology-only
 *        compact counterpart.
 *
 * Three things happen during compaction:
 *
 *   1. **Trim sparse top layers.** Walking from the source's
 *      @c top_occupied_level_id downward, the first layer whose apex
 *      bucket has at least @p min_layer_cap vids becomes the compact
 *      graph's new top. Every vid whose original apex exceeded that
 *      new top is *demoted*: its @c highest_level_id in the compact
 *      graph is reset to the new top, its CSR rows are densely numbered
 *      in the final group, and its neighbor entries for levels
 *      @c [0, new_top] are re-materialized there. Level slots above
 *      the new top are dropped.
 *
 *   2. **Parallel per-vid work.** Both @c VertexInfo writes and the
 *      per-level neighbor transcription are embarrassingly parallel
 *      and driven by @c tbb::parallel_for. Local IDs are
 *      assigned deterministically from the final bucket position.
 *      CSR stores only valid neighbors of actual vertices; source
 *      capacity and source offsets do not determine the target layout.
 *
 *   3. **Centroid-based entry point.** The top-bucket centroid is
 *      computed via @c tbb::parallel_reduce; the vid closest to that
 *      centroid is stashed in the compact graph via
 *      @c set_entry_point_vid and consumed by
 *      @c HierarchicalGraphRouter in place of the sampling seed path.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraphCompactor {

    using vertex_num_t   = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t    = typename IndexTraitsT::vertex_id_t;
    using layer_num_t    = typename IndexTraitsT::layer_num_t;
    using layer_id_t     = typename IndexTraitsT::layer_id_t;
    using vec_ele_t      = typename IndexTraitsT::vec_ele_t;
    using distance_t     = typename IndexTraitsT::distance_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;

    using dynamic        = typename IndexTraitsT::dynamic;
    using compact        = typename IndexTraitsT::compact;

    static constexpr vertex_id_t invalid_vertex_id =
        IndexTraitsT::invalid_vertex_id;

public:
    /** @brief Minimum apex population for a top layer to survive
     *         compaction. Top buckets thinner than this are trimmed
     *         and their vids demoted into the next-lower layer.
     *         Pulled from @c IndexTraitsT so the trim threshold shares
     *         a single source of truth with the r-net capacity floor. */
    static constexpr vertex_num_t min_layer_cap = IndexTraitsT::min_layer_cap;

    /**
     * @brief Parallel-compact @p src into a fresh compact graph.
     *
     * @param src        Dynamic source graph. Must have
     *                   @c assign_layer completed for every vid in
     *                   @c [0, src.get_num_vertices()).
     * @param vecs_data  Base-vector storage, used to compute the
     *                   top-bucket centroid + entry point.
     * @param dist_func  Distance functor used to pick the top-bucket
     *                   vid closest to the centroid.
     * @return A new compact::HierarchicalGraph with a trimmed
     *         hierarchy and a precomputed entry point.
     */
    template <typename DistFuncT>
    static auto compact_graph(
        const typename dynamic::hierarchical_graph_t& src,
        const vector_array_t&                         vecs_data,
        const DistFuncT&                              dist_func
    ) -> typename compact::hierarchical_graph_t {
        using src_graph_t     = typename dynamic::hierarchical_graph_t;
        using compact_graph_t = typename compact::hierarchical_graph_t;

        const vertex_num_t num_vertices    = src.get_num_vertices();
        const vertex_num_t ul_max_nbr_size = src.ul_max_nbr_size();
        const vertex_num_t bl_max_nbr_size = src.bl_max_nbr_size();
        const layer_id_t   src_top         = src.top_occupied_level_id();

        // ---- Empty source: return a degenerate compact graph. ----
        if (num_vertices == 0 || src_top == src_graph_t::invalid_level_id) {
            std::vector<std::size_t> empty_cap(1, 0);
            return compact_graph_t(
                layer_id_t{0}, ul_max_nbr_size, bl_max_nbr_size,
                num_vertices, std::move(empty_cap));
        }

        // =============================================================
        //   Step 1: Determine the new top after trimming.
        // =============================================================
        //
        // Walk src_top → 0; the first bucket with >= min_layer_cap
        // apex vids is the new top. If nothing qualifies, pin to L0.
        layer_id_t new_top = 0;
        for (layer_id_t h = src_top; ; --h) {
            if (src.get_vids_with_highest_level(h).size() >= min_layer_cap) {
                new_top = h;
                break;
            }
            if (h == 0) { new_top = 0; break; }
        }

        // =============================================================
        //   Step 2: Build final per-apex buckets before sizing storage.
        // =============================================================
        // Lower groups preserve source order. The new top contains its
        // original bucket, then demoted buckets in ascending source-layer
        // order. Each assigned vid therefore belongs to one final group.
        const std::size_t num_groups = static_cast<std::size_t>(new_top) + 1;
        std::vector<std::vector<vertex_id_t>> final_buckets(num_groups);
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, num_groups),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t group = r.begin(); group != r.end(); ++group) {
                    auto& bucket = final_buckets[group];
                    const layer_id_t h = static_cast<layer_id_t>(group);
                    const auto& src_bucket = src.get_vids_with_highest_level(h);
                    bucket.assign(src_bucket.begin(), src_bucket.end());
                    if (h == new_top) {
                        for (layer_id_t upper = static_cast<layer_id_t>(new_top + 1);
                             upper <= src_top; ++upper) {
                            const auto& demoted = src.get_vids_with_highest_level(upper);
                            bucket.insert(bucket.end(), demoted.begin(), demoted.end());
                        }
                    }
                }
            });

        // Allocate CSR rows from final bucket sizes, never from source reservations or capacities.
        std::vector<std::size_t> vertex_counts_per_group(num_groups);
        for (std::size_t highest_level = 0; highest_level < num_groups; ++highest_level) {
            vertex_counts_per_group[highest_level] = final_buckets[highest_level].size();
        }
        compact_graph_t result(new_top, ul_max_nbr_size, bl_max_nbr_size,
                               num_vertices, vertex_counts_per_group);
        auto& compact_buckets = result.get_vids_by_highest_level_mut();
        compact_buckets = std::move(final_buckets);
        auto& vertex_info_table = result.get_vertex_info_table_mut();

        // Each bucket position identifies h+1 consecutive rows, ordered h down to zero.
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, num_groups),
            [&](const tbb::blocked_range<std::size_t>& groups) {
                for (std::size_t highest_level = groups.begin(); highest_level != groups.end();
                     ++highest_level) {
                    const auto& bucket = compact_buckets[highest_level];
                    auto row_counts = result.get_nbr_offsets_mut(static_cast<layer_id_t>(highest_level));
                    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, bucket.size()),
                        [&](const tbb::blocked_range<std::size_t>& vertices) {
                            for (std::size_t local_vid = vertices.begin(); local_vid != vertices.end();
                                 ++local_vid) {
                                const vertex_id_t vid = bucket[local_vid];
                                vertex_info_table[vid] = {static_cast<layer_id_t>(highest_level),
                                                          static_cast<vertex_id_t>(local_vid)};
                                for (layer_id_t level_id = 0; level_id <= highest_level; ++level_id) {
                                    std::size_t valid_count = 0;
                                    for (const auto& neighbor : src.fetch_level_nbrs(vid, level_id)) {
                                        if (neighbor.is_invalid()) break;
                                        ++valid_count;
                                    }
                                    const std::size_t row_index = local_vid * (highest_level + 1) +
                                                                  (highest_level - level_id);
                                    row_counts[row_index] = valid_count;
                                }
                            }
                        });
                }
            });
        result.allocate_neighbors_from_row_counts();

        // All destination spans are disjoint. Preserve valid neighbor order and drop trimmed levels.
        tbb::parallel_for(tbb::blocked_range<vertex_num_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_num_t>& vertices) {
                for (vertex_id_t vid = vertices.begin(); vid != vertices.end(); ++vid) {
                    const layer_id_t highest_level = vertex_info_table[vid].highest_level;
                    if (highest_level == compact_graph_t::invalid_level_id) continue;
                    for (layer_id_t level_id = 0; level_id <= highest_level; ++level_id) {
                        const auto source_neighbors = src.fetch_level_nbrs(vid, level_id);
                        auto target_neighbors = result.fetch_level_nbrs_mut(vid, level_id);
                        for (std::size_t neighbor_index = 0; neighbor_index < target_neighbors.size();
                             ++neighbor_index) {
                            target_neighbors[neighbor_index] = source_neighbors[neighbor_index].get_vid();
                        }
                    }
                }
            });

        // =============================================================
        //   Step 6: Centroid-based entry point.
        // =============================================================
        //
        // entry_point = argmin over bucket[new_top] of
        //               dist(vid, centroid_of_bucket_new_top).
        // Parallel reduce over the top bucket for both the centroid
        // accumulation (thread-local double sums) and the argmin.
        const vertex_id_t entry_vid = _compute_entry_point_vid(
            compact_buckets[new_top], vecs_data, dist_func);
        result.set_entry_point_vid(entry_vid);

        return result;
    }

private:
    /**
     * @brief Centroid of the vectors in @p bucket, then argmin distance
     *        back to the bucket.
     *
     * Returns @c invalid_vertex_id iff @p bucket is empty.
     */
    template <typename DistFuncT>
    static auto _compute_entry_point_vid(
        const std::vector<vertex_id_t>& bucket,
        const vector_array_t&           vecs_data,
        const DistFuncT&                dist_func
    ) -> vertex_id_t {
        const std::size_t N = bucket.size();
        if (N == 0) return invalid_vertex_id;

        const std::size_t vec_dim =
            static_cast<std::size_t>(vecs_data.get_vec_dim());

        // ---- Parallel centroid accumulation ----
        // Thread-local double buffers avoid false sharing. The final
        // combine is serial over the (small) number of thread-locals.
        tbb::enumerable_thread_specific<std::vector<double>> local_sums(
            [&]() { return std::vector<double>(vec_dim, 0.0); });

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, N),
            [&](const tbb::blocked_range<std::size_t>& r) {
                auto& local = local_sums.local();
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* v = vecs_data.get(bucket[i]);
                    for (std::size_t d = 0; d < vec_dim; ++d) {
                        local[d] += static_cast<double>(v[d]);
                    }
                }
            });

        std::vector<double> coordinate_sums(vec_dim, 0.0);
        for (const auto& local : local_sums) {
            for (std::size_t d = 0; d < vec_dim; ++d) {
                coordinate_sums[d] += local[d];
            }
        }
        const double inv_N = 1.0 / static_cast<double>(N);
        std::vector<vec_ele_t> centroid(vec_dim);
        for (std::size_t d = 0; d < vec_dim; ++d) {
            centroid[d] = static_cast<vec_ele_t>(coordinate_sums[d] * inv_N);
        }

        // ---- Parallel argmin over the top bucket ----
        struct Best {
            distance_t  dist = std::numeric_limits<distance_t>::max();
            vertex_id_t vid  = invalid_vertex_id;
        };
        const Best best = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, N),
            Best{},
            [&](const tbb::blocked_range<std::size_t>& r, Best init) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const distance_t d = dist_func(
                        centroid.data(), vecs_data.get(bucket[i]));
                    if (d < init.dist) {
                        init.dist = d;
                        init.vid  = bucket[i];
                    }
                }
                return init;
            },
            [](const Best& a, const Best& b) {
                return (a.dist < b.dist) ? a : b;
            });

        return best.vid;
    }

};  // class HierarchicalGraphCompactor

}   // namespace cpu
}   // namespace artea
