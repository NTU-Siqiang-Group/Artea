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
#include <limits>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <artea/common/logger.hpp>

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
 *      graph is reset to the new top, a fresh slot is bump-allocated
 *      in @c arena[new_top], and its neighbor entries for levels
 *      @c [0, new_top] are re-materialized there. Level slots above
 *      the new top are dropped.
 *
 *   2. **Parallel per-vid work.** Both @c VertexInfo writes and the
 *      per-level neighbor transcription are embarrassingly parallel
 *      and driven by @c tbb::parallel_for. Demoted-vid slot offsets
 *      are assigned deterministically from their index in the
 *      demoted-vid list, so no atomics are needed on the hot path.
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

    using vertex_num_t     = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t      = typename IndexTraitsT::vertex_id_t;
    using layer_num_t      = typename IndexTraitsT::layer_num_t;
    using layer_id_t       = typename IndexTraitsT::layer_id_t;
    using vec_ele_t        = typename IndexTraitsT::vec_ele_t;
    using distance_t       = typename IndexTraitsT::distance_t;
    using vector_array_t   = typename IndexTraitsT::vector_array_t;
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;

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
     * @param dataset    Source vector dataset. Provides base storage
     *                   (used to compute the top-bucket centroid +
     *                   entry point) and, for EUCLIDEAN metric, gets
     *                   @c enable_fast_L2 called so we can copy its
     *                   precomputed @c ||p||^2 cache into the compact
     *                   graph for FastL2 search.
     * @param dist_func  Distance functor used to pick the top-bucket
     *                   vid closest to the centroid.
     * @return A new compact::HierarchicalGraph with a trimmed
     *         hierarchy and a precomputed entry point.
     */
    template <typename DistFuncT>
    static auto compact_graph(
        const typename dynamic::hierarchical_graph_t& src,
        vector_dataset_t&                             dataset,
        const DistFuncT&                              dist_func
    ) -> typename compact::hierarchical_graph_t {
        using src_graph_t     = typename dynamic::hierarchical_graph_t;
        using compact_graph_t = typename compact::hierarchical_graph_t;

        const vector_array_t& vecs_data = dataset.get_base_vecs();

        const vertex_num_t num_vertices    = src.get_num_vertices();
        const vertex_num_t ul_max_nbr_size = src.ul_max_nbr_size();
        const vertex_num_t bl_max_nbr_size = src.bl_max_nbr_size();
        const layer_id_t   src_top         = src.top_occupied_level_id();

        // ---- Empty source: return a degenerate compact graph. ----
        if (num_vertices == 0 || src_top == src_graph_t::unassigned_highest_level_id) {
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
        //   Step 2: Flatten the demoted-vid list.
        // =============================================================
        //
        // Every vid whose original apex is strictly greater than
        // new_top is demoted to new_top. Their compact slots are
        // appended after the source's existing arena[new_top] layout,
        // and their VertexInfo is rewritten to point there.
        std::vector<vertex_id_t> demoted_vids;
        for (layer_id_t h = static_cast<layer_id_t>(new_top + 1);
             h <= src_top; ++h)
        {
            const auto& src_upper_bucket =
                src.get_vids_with_highest_level(h);
            demoted_vids.insert(demoted_vids.end(),
                                src_upper_bucket.begin(),
                                src_upper_bucket.end());
        }
        const std::size_t demoted_count = demoted_vids.size();

        // =============================================================
        //   Step 3: Size + allocate the compact arenas.
        // =============================================================
        //
        // Arenas 0..new_top-1 keep the source's capacity verbatim.
        // arena[new_top] grows by demoted_count slots to hold the
        // demoted vids' re-materialized neighbor rows.
        std::vector<std::size_t> arena_vid_capacity(
            static_cast<std::size_t>(new_top) + 1);
        for (layer_id_t h = 0; h <= new_top; ++h) {
            // Upper levels 1..h each use ul_max_nbr_size; L0 uses bl.
            const std::size_t slot_nbrs_count =
                static_cast<std::size_t>(h) *
                    static_cast<std::size_t>(ul_max_nbr_size)
                + static_cast<std::size_t>(bl_max_nbr_size);
            std::size_t slot_capacity =
                static_cast<std::size_t>(src.get_arena_capacity_in_arena(h));
            if (h == new_top) slot_capacity += demoted_count;
            arena_vid_capacity[h] = slot_capacity * slot_nbrs_count;
        }

        compact_graph_t result(
            new_top, ul_max_nbr_size, bl_max_nbr_size,
            num_vertices, std::move(arena_vid_capacity));

        // =============================================================
        //   Step 4: Populate VertexInfo (parallel).
        // =============================================================
        //
        // Two disjoint parallel passes:
        //   (a) Non-demoted vids (H_src <= new_top): copy
        //       (H_src, src slot_offset) verbatim.
        //   (b) Demoted vids (H_src > new_top): rewrite to
        //       (new_top, demoted_base_offset + i * top_slot_nbrs_count)
        //       where i is the vid's index in demoted_vids. No
        //       atomics needed — offsets are a pure function of i.
        auto& compact_vit = result.get_vertex_info_table_mut();

        const std::size_t top_slot_nbrs_count =
            static_cast<std::size_t>(new_top) *
                static_cast<std::size_t>(ul_max_nbr_size)
            + static_cast<std::size_t>(bl_max_nbr_size);
        const std::size_t demoted_base_offset =
            static_cast<std::size_t>(
                src.get_arena_capacity_in_arena(new_top)) *
            top_slot_nbrs_count;

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const layer_id_t H_src = src.get_highest_level_id(vid);
                    if (H_src <= new_top) {
                        compact_vit[vid].highest_level_id = H_src;
                        compact_vit[vid].slot_offset      =
                            src.get_slot_offset(vid);
                    }
                    // Demoted vids are filled in the next loop below.
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, demoted_count),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t vid = demoted_vids[i];
                    compact_vit[vid].highest_level_id = new_top;
                    compact_vit[vid].slot_offset      =
                        demoted_base_offset + i * top_slot_nbrs_count;
                }
            });

        // =============================================================
        //   Step 5: Rebuild per-apex buckets.
        // =============================================================
        //
        // Buckets h < new_top are bulk-copied from the source
        // (parallel per bucket). bucket[new_top] is the source's own
        // top bucket plus every demoted vid (whose apex has just been
        // rewritten to new_top in Step 4).
        auto& compact_buckets = result.get_vids_by_highest_level_mut();

        if (new_top > 0) {
            tbb::parallel_for(
                tbb::blocked_range<layer_id_t>(0, new_top),
                [&](const tbb::blocked_range<layer_id_t>& r) {
                    for (layer_id_t h = r.begin(); h != r.end(); ++h) {
                        const auto& src_bucket =
                            src.get_vids_with_highest_level(h);
                        compact_buckets[h].reserve(src_bucket.size());
                        for (const vertex_id_t vid : src_bucket) {
                            compact_buckets[h].push_back(vid);
                        }
                    }
                });
        }

        {
            const auto& src_top_bucket =
                src.get_vids_with_highest_level(new_top);
            compact_buckets[new_top].reserve(
                src_top_bucket.size() + demoted_count);
            for (const vertex_id_t vid : src_top_bucket) {
                compact_buckets[new_top].push_back(vid);
            }
            for (const vertex_id_t vid : demoted_vids) {
                compact_buckets[new_top].push_back(vid);
            }
        }

        // =============================================================
        //   Step 6: Transcribe per-level neighbors (parallel per vid).
        // =============================================================
        //
        // Every compact VertexInfo is now final. For each vid:
        //
        //   for cur_level in [0, H_new]:
        //     copy src.fetch_layer_nbrs(vid, cur_level) → compact slot
        //
        // Non-demoted vids: H_new == H_src, same layout on both sides,
        // essentially a memcpy per level row.
        // Demoted vids: H_new == new_top, so the compact slot is
        // shorter; we re-materialize only levels 0..new_top and drop
        // the upper level rows.
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const auto& vinfo = compact_vit[vid];
                    const layer_id_t H_new = vinfo.highest_level_id;
                    if (H_new == compact_graph_t::unassigned_highest_level_id) {
                        continue;
                    }
                    vertex_id_t* slot_base =
                        result.arena_base(H_new) + vinfo.slot_offset;

                    for (layer_id_t cur_level = 0;
                         cur_level <= H_new;
                         ++cur_level)
                    {
                        const auto src_nbrs =
                            src.fetch_layer_nbrs(vid, cur_level);
                        const std::size_t level_offset =
                            static_cast<std::size_t>(H_new - cur_level) *
                            static_cast<std::size_t>(ul_max_nbr_size);
                        vertex_id_t* dst = slot_base + level_offset;

                        std::size_t out = 0;
                        for (const auto& nbr : src_nbrs) {
                            if (nbr.is_invalid()) break;
                            dst[out++] = nbr.get_vid();
                        }
                        // Trailing sentinels were pre-filled by the
                        // compact graph's constructor.
                    }
                }
            });

        // =============================================================
        //   Step 7: Centroid-based entry point.
        // =============================================================
        //
        // entry_point = argmin over bucket[new_top] of
        //               dist(vid, centroid_of_bucket_new_top).
        // Parallel reduce over the top bucket for both the centroid
        // accumulation (thread-local double sums) and the argmin.
        const vertex_id_t entry_vid = _compute_entry_point_vid(
            compact_buckets[new_top], vecs_data, dist_func);
        result.set_entry_point_vid(entry_vid);

        // =============================================================
        //   Step 8: FastL2 norm cache (EUCLIDEAN only).
        // =============================================================
        //
        // For Euclidean search the compact-mode router uses
        // SIMDDistance::fast_euclidean, which needs per-base ||p||^2.
        // Trigger the dataset's parallel norm computation and copy the
        // result into the compact graph so it's lifetime-independent of
        // the source dataset.
        if constexpr (DistFuncT::distance_metrics ==
                      DistanceMetricsT::EUCLIDEAN)
        {
            dataset.enable_fast_L2();
            result.set_base_norms(dataset.get_base_norms());
        }

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

        std::vector<double> centroid_d(vec_dim, 0.0);
        for (const auto& local : local_sums) {
            for (std::size_t d = 0; d < vec_dim; ++d) {
                centroid_d[d] += local[d];
            }
        }
        const double inv_N = 1.0 / static_cast<double>(N);
        std::vector<vec_ele_t> centroid(vec_dim);
        for (std::size_t d = 0; d < vec_dim; ++d) {
            centroid[d] = static_cast<vec_ele_t>(centroid_d[d] * inv_N);
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
