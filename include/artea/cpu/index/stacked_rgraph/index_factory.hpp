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
 * @FilePath: /Artea/include/artea/cpu/index/stacked_rgraph/index_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Insertion engine for stacked_rgraph::IndexStructure.
 *
 * Algorithm outline (per-vertex):
 *   Step A — Descent: beam_search from the current top occupied level
 *            down to level 1 (never level 0). Each level's queue is
 *            cloned so later select-neighbor searches can resume from
 *            it with a larger capacity.
 *   Step B — Compute highest_insert_level_id via the r-net covering
 *            rule.
 *   Step C — Claim a slot in the hierarchical graph via assign_layer.
 *   Step D — Edge insertion:
 *            - Run one select-neighbors search at level 1 to build
 *              pruned_l1_nbrs (used for every vertex).
 *            - Level 0 forward: write pruned_l1_nbrs into vid's L0 slot
 *              (first max_nbr_size positions; L0 has 2 * max_nbr_size
 *              capacity, rear half reserved for future refiner passes).
 *            - Level 0 reverse: add vid to each pruned neighbor's L0
 *              slot.
 *            - Level 1 forward + reverse (only if highest_insert_level
 *              >= 1).
 *            - For cur_level in [2, highest_insert_level]: normal
 *              select + forward + reverse.
 *
 * We **never** run beam_search at level 0 (level 0 is the full
 * N-vertex base, searching it during every insertion would be far too
 * expensive).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Insertion algorithm for stacked_rgraph::IndexStructure.
 *
 * Static entry point:
 * @code
 *   IndexFactory::add_vertices(index, std::move(batch_vecs),
 *                              dist_func, pruning_updater);
 * @endcode
 *
 * @tparam GraphFactoryTraitsT The graph-factory traits type.
 */
template <typename GraphFactoryTraitsT>
class IndexFactory {

    using this_index_t     = typename GraphFactoryTraitsT::stacked_rgraph::index_t;

    using vertex_num_t     = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t      = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_num_t      = typename GraphFactoryTraitsT::layer_num_t;
    using layer_id_t       = typename GraphFactoryTraitsT::layer_id_t;
    using distance_t       = typename GraphFactoryTraitsT::distance_t;
    using vec_ele_t        = typename GraphFactoryTraitsT::vec_ele_t;
    using nbr_t            = typename GraphFactoryTraitsT::nbr_t;
    using vector_array_t   = typename GraphFactoryTraitsT::vector_array_t;
    using dist_func_t      = typename GraphFactoryTraitsT::dist_func_t;

    using hierarchical_graph_t =
        typename GraphFactoryTraitsT::dynamic::hierarchical_graph_t;

    using visited_table_t         = typename GraphFactoryTraitsT::visited_table_t;
    using candidate_entry_t       = typename GraphFactoryTraitsT::candidate_entry_t;
    using std_candidate_queue_t   = typename GraphFactoryTraitsT::std_candidate_queue_t;
    using knn_results_t           = typename GraphFactoryTraitsT::knn_results_t;
    using hg_router_t             =
        typename GraphFactoryTraitsT::dynamic::hierarchical_graph_router_t;

    static constexpr vertex_id_t invalid_vertex_id = GraphFactoryTraitsT::invalid_vertex_id;
    static constexpr distance_t  max_distance      = GraphFactoryTraitsT::max_distance;

    static constexpr layer_id_t  unassigned_highest_level_id =
        hierarchical_graph_t::unassigned_highest_level_id;

public:
    /** @brief Number of vertices inserted serially during bootstrap
     *         before switching to @c tbb::parallel_for. */
    static constexpr vertex_num_t startup_points = 500;

    /**
     * @brief Append @p batch_vecs to @p index's owned storage, then
     *        insert every newly-appended vector as a new vertex.
     *
     * @param index            The index whose hierarchy is being grown.
     * @param batch_vecs       Batch to insert. Moved into @p index.
     * @param dist_func        Distance functor (must outlive this call).
     * @param pruning_updater  Neighbor-pruning object implementing
     *        @c update_impl(pivot_vid, std::vector<nbr_t>&, max_nbr_size).
     *        Used for both forward-edge pruning and reverse-edge
     *        overflow handling.
     */
    template <typename PruningUpdaterT>
    static auto add_vertices(
        this_index_t&        index,
        vector_array_t&&     batch_vecs,
        const dist_func_t&   dist_func,
        PruningUpdaterT&     pruning_updater
    ) -> void {
        const vertex_num_t batch_size =
            static_cast<vertex_num_t>(batch_vecs.get_num_vecs());
        if (batch_size == 0) return;

        index.append_vecs(std::move(batch_vecs));
        const vertex_id_t first_new_vid = index.add_vertices(batch_size);

        const auto& vecs_storage = index.get_vecs_storage();
        const vertex_num_t total_vecs =
            static_cast<vertex_num_t>(vecs_storage.get_num_vecs());

        hg_router_t router(
            vecs_storage, dist_func,
            /*topk=*/std::max(index.search_nn_qs(), index.select_nbrs_qs()),
            /*search_nn_qs=*/index.search_nn_qs(),
            /*candidate_queue_size=*/index.select_nbrs_qs());

        tbb::enumerable_thread_specific<visited_table_t> visited_pool(
            [total_vecs]() {
                return visited_table_t(static_cast<std::size_t>(total_vecs));
            });

        const vertex_id_t serial_cutoff =
            first_new_vid +
            std::min<vertex_num_t>(startup_points, batch_size);

        {
            auto& visited = visited_pool.local();
            for (vertex_id_t vid = first_new_vid; vid < serial_cutoff; ++vid) {
                _insert_one(index, router, vid, dist_func,
                            pruning_updater, visited);
            }
        }
        if (serial_cutoff == first_new_vid + batch_size) return;

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(
                serial_cutoff, first_new_vid + batch_size),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                auto& visited = visited_pool.local();
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    _insert_one(index, router, vid, dist_func,
                                pruning_updater, visited);
                }
            }
        );
    }

private:
    // -----------------------------------------------------------------
    //   Per-vertex insertion
    // -----------------------------------------------------------------

    template <typename PruningUpdaterT>
    static auto _insert_one(
        this_index_t&       index,
        const hg_router_t&  router,
        const vertex_id_t   new_vid,
        const dist_func_t&  /*dist_func*/,
        PruningUpdaterT&    pruning_updater,
        visited_table_t&    visited
    ) -> void {
        const vec_ele_t* new_vec        = index.get_vecs_storage().get(new_vid);
        const vertex_num_t search_nn_qs   = index.search_nn_qs();
        const vertex_num_t select_nbrs_qs = index.select_nbrs_qs();
        const layer_num_t  max_restrict_level = index.max_restrict_level();
        const layer_id_t   top_level_id = index.top_occupied_highest_level_id();

        // ==============================================================
        //   Step A — Descent: top_level_id → level 1 (never level 0)
        // ==============================================================
        std::vector<distance_t> min_dist_per_level;
        std::vector<std::unique_ptr<std_candidate_queue_t>> queue_snapshot_per_level;

        if (top_level_id != unassigned_highest_level_id && top_level_id >= 1) {
            min_dist_per_level.assign(static_cast<std::size_t>(top_level_id) + 1, max_distance);
            queue_snapshot_per_level.resize(static_cast<std::size_t>(top_level_id) + 1);
            std_candidate_queue_t descent_queue(search_nn_qs);
            router.sample_entries(index, top_level_id, new_vec, descent_queue);

            for (layer_id_t cur_level_id = top_level_id; cur_level_id >= 1; --cur_level_id) {
                router.beam_search(new_vec, index, cur_level_id, descent_queue, visited);
                if (descent_queue.get_result_size() > 0) {
                    min_dist_per_level[cur_level_id] = descent_queue.best_result_distance();
                    queue_snapshot_per_level[cur_level_id] = std::make_unique<std_candidate_queue_t>(descent_queue.clone());
                }
                if (cur_level_id == 1) break;   // avoid layer_id_t underflow
            }
        }

        // ==============================================================
        //   Step B — Compute highest_insert_level_id
        // ==============================================================
        //
        // r-net rule: at each layer h in [1, max_restrict_level],
        // measure q's nearest neighbor distance WITHIN L_h. If that
        // distance < R_h, q is "absorbed" at L_h and we move up to
        // check the next coarser layer. If distance >= R_h, q cannot
        // be absorbed at L_h. q's insertion level is the smallest
        // (bottom-most) layer where q is not absorbed. If every layer
        // up to max_restrict_level absorbs q, it stays at the base only
        // (highest_insert_level_id = 0).
        //
        // min_dist_per_level[h] approximates NN_dist_in_L_h: the
        // descent queue at cur=h contains only L_h participants
        // (neighbors reached via fetch_layer_nbrs(_, h) all have
        // highest_level_id >= h), so min over the queue is bounded
        // above by the true NN in L_h. When h > top_level_id, L_h has
        // no participants yet and q is trivially not absorbed.
        layer_id_t highest_insert_level_id = 0;
        for (layer_id_t h = 1; h <= static_cast<layer_id_t>(max_restrict_level); ++h) {
            const distance_t nn_at_h = (static_cast<std::size_t>(h) < min_dist_per_level.size())
                ? min_dist_per_level[h] : max_distance;
            if (nn_at_h >= index.radius_at(h)) {
                highest_insert_level_id = h;
                break;
            }
        }

        // ==============================================================
        //   Step C — Assign slot
        // ==============================================================
        index.assign_layer(new_vid, highest_insert_level_id);

        // ==============================================================
        //   Step D — Edge insertion
        // ==============================================================
        //
        // Always run select-neighbors search at level 1 (never at 0).
        // Its pruned result is used twice:
        //   - As the L0 forward + reverse edges (no L0 beam_search).
        //   - As the L1 forward + reverse edges (if vid participates
        //     at L1, i.e. highest_insert_level_id >= 1).
        // For cur_level >= 2, run the normal per-level loop.

        auto run_select_at_level = [&](const layer_id_t target_level_id)
            -> std::vector<nbr_t>
        {
            std_candidate_queue_t* select_queue_ptr = nullptr;
            std::unique_ptr<std_candidate_queue_t> select_queue_owner;

            if (target_level_id < queue_snapshot_per_level.size() && queue_snapshot_per_level[target_level_id]) {
                // Reuse the descent snapshot; bump its capacity from
                // search_nn_qs (L1) to select_nbrs_qs (L2), then
                // continue the beam search for the extra width.
                select_queue_owner = std::move(queue_snapshot_per_level[target_level_id]);
                select_queue_owner->set_capacity(select_nbrs_qs);
                select_queue_ptr = select_queue_owner.get();
            } else {
                // No snapshot (e.g. first-ever insertion, or target
                // level exceeds top_level_id) — fresh random seeds.
                select_queue_owner = std::make_unique<std_candidate_queue_t>(select_nbrs_qs);
                router.sample_entries(index, target_level_id, new_vec, *select_queue_owner);
                select_queue_ptr = select_queue_owner.get();
            }
            router.beam_search(new_vec, index, target_level_id, *select_queue_ptr, visited);

            std::vector<nbr_t> pruned_results;
            const std::size_t result_size = select_queue_ptr->get_result_size();
            if (result_size == 0) return pruned_results;

            auto sorted = select_queue_ptr->extract_results(result_size);
            pruned_results.reserve(sorted.size());
            for (const auto& cand : sorted) {
                pruned_results.emplace_back(cand.get_base_vid(), cand.get_distance(), /*is_new=*/true);
            }
            pruning_updater.update_impl(new_vid, pruned_results, index.max_nbr_size(target_level_id));
            return pruned_results;
        };

        auto write_forward_edges = [&](const layer_id_t target_level_id,
                                       const std::vector<nbr_t>& pruned_results,
                                       const std::size_t max_write_count)
        {
            index.with_locked_nbrs(new_vid, target_level_id,
                [&](std::span<nbr_t> slot, vertex_num_t /*cnt == 0*/) {
                    const std::size_t write_count = std::min(
                        std::min(pruned_results.size(), slot.size()),
                        max_write_count);
                    for (std::size_t i = 0; i < write_count; ++i) {
                        slot[i] = pruned_results[i];
                    }
                    if (write_count < slot.size()) {
                        slot[write_count] = nbr_t::make_invalid_nbr();
                    }
                });
        };

        auto write_reverse_edges = [&](const layer_id_t target_level_id,
                                       const std::vector<nbr_t>& pruned_results)
        {
            for (std::size_t i = 0; i < pruned_results.size(); ++i) {
                const vertex_id_t nbr_vid  = pruned_results[i].get_vid();
                const distance_t  nbr_dist = pruned_results[i].get_distance();
                const nbr_t new_reverse_nbr =
                    nbr_t::make_new_nbr(new_vid, nbr_dist);

                index.with_locked_nbrs(nbr_vid, target_level_id,
                    [&](std::span<nbr_t> slot, vertex_num_t cnt) {
                        const vertex_num_t slot_cap =
                            static_cast<vertex_num_t>(slot.size());
                        if (cnt < slot_cap) {
                            slot[cnt] = new_reverse_nbr;
                            if (cnt + 1 < slot_cap) {
                                slot[cnt + 1] = nbr_t::make_invalid_nbr();
                            }
                            return;
                        }
                        // Slot is full — re-run RNG pruning on
                        // existing slot ∪ {new_reverse_nbr}.
                        std::vector<nbr_t> merged;
                        merged.reserve(slot_cap + 1);
                        for (vertex_num_t k = 0; k < slot_cap; ++k) {
                            merged.push_back(slot[k]);
                        }
                        merged.push_back(new_reverse_nbr);
                        std::sort(merged.begin(), merged.end(),
                            [](const nbr_t& a, const nbr_t& b) {
                                return a.get_distance() < b.get_distance();
                            });
                        pruning_updater.update_impl(
                            nbr_vid, merged, slot_cap);
                        for (std::size_t k = 0;
                             k < merged.size() && k < slot_cap; ++k)
                        {
                            slot[k] = merged[k];
                        }
                        if (merged.size() < slot_cap) {
                            slot[merged.size()] = nbr_t::make_invalid_nbr();
                        }
                    });
            }
        };

        // ---- cur_level_id == 1: always run select (drives L0 + L1) ----
        const std::vector<nbr_t> pruned_l1_nbrs = run_select_at_level(1);

        if (!pruned_l1_nbrs.empty()) {
            // L0 forward: write the pruned L1 neighbors into the first
            // max_nbr_size positions of vid's L0 slot (capacity is
            // 2 * max_nbr_size; the back half stays as invalid sentinel
            // for future refiner passes).
            write_forward_edges(
                /*target_level_id=*/0,
                pruned_l1_nbrs,
                /*max_write_count=*/index.max_nbr_size());

            // L0 reverse: add vid to each pruned neighbor's L0 slot.
            write_reverse_edges(/*target_level_id=*/0, pruned_l1_nbrs);

            if (highest_insert_level_id >= 1) {
                // L1 forward + reverse using the same pruned set.
                write_forward_edges(
                    1, pruned_l1_nbrs,
                    /*max_write_count=*/index.max_nbr_size(1));
                write_reverse_edges(1, pruned_l1_nbrs);
            }
        }

        // ---- cur_level_id >= 2: normal forward + reverse ----
        for (layer_id_t cur_level_id = 2;
             cur_level_id <= highest_insert_level_id;
             ++cur_level_id)
        {
            const std::vector<nbr_t> pruned_results =
                run_select_at_level(cur_level_id);
            if (pruned_results.empty()) continue;
            write_forward_edges(
                cur_level_id, pruned_results,
                /*max_write_count=*/index.max_nbr_size(cur_level_id));
            write_reverse_edges(cur_level_id, pruned_results);
        }
    }

};  // class IndexFactory

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
