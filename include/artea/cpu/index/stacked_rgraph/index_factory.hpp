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
 *   Step D — Edge insertion for cur_level_id in
 *            [0, highest_insert_level_id]: per-level select_neighbors +
 *            forward + reverse.
 *            L0 doesn't get its own descent pass (running beam_search
 *            on the full N-vertex base every insert would be far too
 *            expensive), so before the loop we seed
 *            descent_queue_per_level[0] from descent_queue_per_level[1]
 *            via seed_from_queue. The L0 select then expands from
 *            those L1 NN seeds with the standard select_nbrs_qs beam
 *            (default 100).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
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
    using ratio_t          = typename GraphFactoryTraitsT::ratio_t;

    using hierarchical_pruning_updater_t =
        typename GraphFactoryTraitsT::hierarchical_pruning_updater_t;

    using hierarchical_graph_t =
        typename GraphFactoryTraitsT::dynamic::hierarchical_graph_t;

    using visited_table_t         = typename GraphFactoryTraitsT::visited_table_t;
    using visited_table_pool_t    = typename GraphFactoryTraitsT::visited_table_pool_t;
    using candidate_entry_t       = typename GraphFactoryTraitsT::candidate_entry_t;
    using std_candidate_queue_t   = typename GraphFactoryTraitsT::std_candidate_queue_t;
    using knn_results_t           = typename GraphFactoryTraitsT::knn_results_t;
    using hg_router_t             =
        typename GraphFactoryTraitsT::hierarchical_graph_router_t;
    using candidate_sample_utils_t =
        typename GraphFactoryTraitsT::candidate_sample_utils_t;

    static constexpr vertex_id_t invalid_vertex_id = GraphFactoryTraitsT::invalid_vertex_id;
    static constexpr distance_t  max_distance      = GraphFactoryTraitsT::max_distance;

    static constexpr layer_id_t  unassigned_highest_level_id =
        hierarchical_graph_t::unassigned_highest_level_id;

public:
    /** @brief Number of vertices inserted serially during bootstrap
     *         before switching to @c tbb::parallel_for. */
    static constexpr vertex_num_t startup_points = 0;

    /**
     * @brief Append @p batch_vecs to @p index's owned storage, then
     *        insert every newly-appended vector as a new vertex. The
     *        RNG pruning coefficients are taken from
     *        @c index.pruning_config().
     *
     * @param index         The index whose hierarchy is being grown.
     * @param batch_vecs    Batch to insert. Moved into @p index.
     * @param dist_func     Distance functor (must outlive this call).
     * @param insert_on_L0  When true (default), each new vertex's L0
     *                      neighbors are selected and written as usual.
     *                      When false, L0 edge construction is skipped
     *                      entirely (slot left empty, seed-from-L1 step
     *                      elided). Upper levels still get their edges.
     *                      Used by the artea_graph pipeline where L0 is
     *                      rebuilt from scratch by per-layer refinement,
     *                      so spending insertion-time on L0 neighbors
     *                      would be wasted work.
     * @param shuffle_insertion_order When true, the order in which new
     *                      vids are fed into @c _insert_one is a
     *                      deterministic shuffle of the batch (storage
     *                      layout / vid assignment unchanged). This
     *                      breaks the coupling between dataset ordering
     *                      and time-locality of insertions, which
     *                      otherwise biases the hierarchy on
     *                      cluster-sorted datasets. Default false
     *                      preserves the legacy dataset-order build.
     */
    static auto add_vertices(
        this_index_t&      index,
        vector_array_t&&   batch_vecs,
        const dist_func_t& dist_func,
        const bool         insert_on_L0 = true,
        const bool         shuffle_insertion_order = false
    ) -> void {
        const vertex_num_t batch_size = static_cast<vertex_num_t>(batch_vecs.get_num_vecs());
        if (batch_size == 0) return;

        index.append_vecs(std::move(batch_vecs));
        const vertex_id_t first_new_vid = index.add_vertices(batch_size);

        // Permutation of the new-vid range. Walked in its natural order
        // (identity) when shuffle is off; Fisher-Yates-shuffled with a
        // fixed seed when on, so two runs on the same dataset produce
        // the same graph.
        std::vector<vertex_id_t> insert_order(batch_size);
        std::iota(insert_order.begin(), insert_order.end(), first_new_vid);
        if (shuffle_insertion_order) {
            std::mt19937_64 rng(/*seed=*/0xA17EA5EEDULL);
            std::shuffle(insert_order.begin(), insert_order.end(), rng);
        }

        // Construct the pruning updater AFTER append_vecs so the storage
        // reference it captures already points at populated data (belt-
        // and-suspenders; vecs_storage_t is stable either way). This
        // build-time pruner applies the plain RNG rule (threshold =
        // ori_dist) at every level — scale/shift are a post-refining-
        // only concern now.
        hierarchical_pruning_updater_t pruning_updater(
            dist_func,
            index.get_vecs_storage());

        const auto& vecs_storage = index.get_vecs_storage();
        const vertex_num_t total_vecs = static_cast<vertex_num_t>(vecs_storage.get_num_vecs());

        hg_router_t router(
            vecs_storage, dist_func,
            /*topk=*/std::max(index.search_nn_qs(), index.select_nbrs_qs()),
            /*candidate_queue_size=*/index.select_nbrs_qs());

        visited_table_pool_t visited_pool(total_vecs);

        // Serial bootstrap iterates the permutation prefix; parallel
        // phase iterates the suffix. Both dereference insert_order[i]
        // to get the actual vid.
        const vertex_num_t serial_count =
            std::min<vertex_num_t>(startup_points, batch_size);
        for (vertex_num_t i = 0; i < serial_count; ++i) {
            const vertex_id_t vid = insert_order[i];
            auto& visited = visited_pool.acquire();
            _insert_one(index, router, vid, dist_func,
                        pruning_updater, visited, insert_on_L0);
        }
        if (serial_count == batch_size) return;

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(serial_count, batch_size),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t vid = insert_order[i];
                    // acquire() per-vertex: each insertion starts with a
                    // fresh visited table, shared across its descent +
                    // select phases.
                    auto& visited = visited_pool.acquire();
                    _insert_one(index, router, vid, dist_func,
                                pruning_updater, visited, insert_on_L0);
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
        const dist_func_t&  dist_func,
        PruningUpdaterT&    pruning_updater,
        visited_table_t&    visited,
        const bool          insert_on_L0
    ) -> void {
        const vec_ele_t* new_vec        = index.get_vecs_storage().get(new_vid);
        const vertex_num_t search_nn_qs   = index.search_nn_qs();
        const vertex_num_t select_nbrs_qs = index.select_nbrs_qs();
        const layer_num_t  max_restrict_level = index.max_restrict_level();
        const layer_id_t   top_level_id = index.top_occupied_level_id();

        // ==============================================================
        //   Step A — Descent: top_level_id → level 1 (not including level 0)
        // ==============================================================
        const std::size_t num_cached_queues = static_cast<std::size_t>(max_restrict_level) + 1;
        std::vector<distance_t> min_dist_per_level(num_cached_queues, max_distance);
        std::vector<std::optional<std_candidate_queue_t>> descent_queue_per_level(num_cached_queues);
        for (auto& slot : descent_queue_per_level) { slot.emplace(search_nn_qs); }

        if (top_level_id != unassigned_highest_level_id && top_level_id >= 1) {
            std_candidate_queue_t cur_queue(search_nn_qs);
            candidate_sample_utils_t::sample_single_entry(index.get_vecs_storage(), dist_func, index, new_vec, cur_queue);

            for (layer_id_t cur_level_id = top_level_id; cur_level_id >= 1; --cur_level_id) {
                router.beam_search(new_vec, index, cur_level_id, cur_queue, visited);

                if (cur_queue.get_result_size() > 0) {
                    min_dist_per_level[cur_level_id] = cur_queue.best_result_distance();
                }

                if (cur_level_id == 1) {
                    // Last level: just stash for Step D.
                    descent_queue_per_level[cur_level_id].emplace(std::move(cur_queue));
                    break;
                }

                // Fork a fresh queue for the next (lower) level BEFORE
                // handing ownership of cur_queue to the per-level slot.
                // seed_from_queue copies this layer's top_candidates into
                // the new queue in one O(L) pass with a clean
                // _lower_bound (no cross-layer gating).
                std_candidate_queue_t next_queue(search_nn_qs);
                next_queue.seed_from_queue(cur_queue);
                descent_queue_per_level[cur_level_id].emplace(std::move(cur_queue));
                cur_queue = std::move(next_queue);
            }
        }

        // Descent's L0 tail: L0 has no beam_search pass of its own, so
        // seed descent_queue_per_level[0] from descent_queue_per_level[1]
        // — those L1 NNs are the closest seeds available without paying
        // the cost of a full L0 beam search during descent.
        //
        // First-vertex bootstrap (top_level_id == unassigned) leaves
        // descent_queue_per_level[1] empty; the seed call is then a
        // no-op and L0's run_select_at_level will safely no-op too.
        //
        // When insert_on_L0 == false, L0 edge construction is skipped
        // entirely (see Step D loop), so the L1→L0 seed transfer is
        // wasted work and we elide it here.
        if (insert_on_L0 && descent_queue_per_level[1] &&
            descent_queue_per_level[1]->get_result_size() > 0) {
            descent_queue_per_level[0]->seed_from_queue(*descent_queue_per_level[1]);
        }

        // ==============================================================
        //   Step B — Compute highest_insert_level_id (match legacy)
        // ==============================================================
        layer_id_t highest_insert_level_id;
        if (top_level_id == unassigned_highest_level_id) {
            // Empty hierarchy: first vertex seeds L_1.
            highest_insert_level_id = 1;
        } else {
            // Default: grow hierarchy by ONE level (capped at
            // max_restrict_level).
            highest_insert_level_id = std::min<layer_id_t>(
                static_cast<layer_id_t>(top_level_id) + 1,
                static_cast<layer_id_t>(max_restrict_level));
            // Override: smallest h where q is absorbed at L_h means
            // q joins only L_0..L_{h-1}.
            for (layer_id_t h = 1; h <= top_level_id; ++h) {
                if (min_dist_per_level[h] <= index.radius_at(h)) {
                    highest_insert_level_id = static_cast<layer_id_t>(h - 1);
                    break;
                }
            }
        }

        // ==============================================================
        //   Step C — Assign slot
        // ==============================================================
        index.assign_layer(new_vid, highest_insert_level_id);

        // ==============================================================
        //   Step D — Edge insertion (now includes L0)
        // ==============================================================
        // For cur_level_id in [start_level_id, highest_insert_level_id],
        // run per-level select + forward + reverse.
        auto run_select_at_level = [&](const layer_id_t target_level_id) -> std::vector<nbr_t> {
            // descent_queue_per_level is pre-sized to max_restrict_level+1
            // with empty queues in every slot (see Step A). Uncached
            // levels therefore present an empty queue here, which makes
            // beam_search a no-op and yields an empty pruned_results
            // that Step D safely skips.
            auto& cached_queue = descent_queue_per_level[target_level_id];
            std_candidate_queue_t select_queue = std::move(*cached_queue);
            cached_queue.reset();
            // Grow the queue to the select-phase beam width and relax
            // the rejection threshold so the wider search can accept
            // candidates that were filtered by the narrower descent pass.
            select_queue.set_capacity(select_nbrs_qs);
            select_queue.reset_lower_bound();
            router.beam_search(new_vec, index, target_level_id, select_queue, visited);

            std::vector<nbr_t> pruned_results;
            const std::size_t result_size = select_queue.get_result_size();
            if (result_size == 0) return pruned_results;

            auto sorted_results = select_queue.extract_results(result_size);
            pruned_results.reserve(sorted_results.size());
            for (const auto& cand : sorted_results) {
                // Skip self: assign_layer already placed new_vid in its
                // bucket, so sample_entries / beam_search may have
                // picked it as a seed with distance 0. Including it
                // here would produce a forward self-loop.
                if (cand.get_vid() == new_vid) continue;
                pruned_results.emplace_back(cand.get_vid(), cand.get_distance(), /*is_new=*/true);
            }
            pruning_updater.update_impl(pruned_results, index.max_nbr_size(target_level_id));
            return pruned_results;
        };

        // ---- cur_level_id in [start_level_id, highest_insert_level_id]:
        //      per-level select + forward + reverse ----
        //
        // start_level_id = 1 when insert_on_L0 == false so the L0 slot is
        // left untouched (refinement-time responsibility). Upper levels
        // are always built regardless of the flag.
        const layer_id_t start_level_id = insert_on_L0 ? layer_id_t{0} : layer_id_t{1};
        for (layer_id_t cur_level_id = start_level_id; cur_level_id <= highest_insert_level_id; ++cur_level_id) {
            const std::vector<nbr_t> pruned_results = run_select_at_level(cur_level_id);
            if (pruned_results.empty()) continue;
            _write_forward_edges(
                index, new_vid, cur_level_id, pruned_results,
                /*max_write_count=*/index.max_nbr_size(cur_level_id));
            _write_reverse_edges(
                index, new_vid, cur_level_id, pruned_results,
                pruning_updater);
        }
    }

    // -----------------------------------------------------------------
    //   Per-level edge writers
    // -----------------------------------------------------------------

    /**
     * @brief Write forward edges for @p new_vid at @p target_level_id.
     *
     * Precondition: the slot for @p new_vid at this level is empty
     * (freshly claimed by @c assign_layer, initialized to invalid
     * sentinels). Writes up to
     * @c min(pruned_results.size(), slot.size(), max_write_count)
     * entries and terminates with an invalid sentinel if any tail
     * remains.
     */
    __attribute__((always_inline))
    static auto _write_forward_edges(
        this_index_t&              index,
        const vertex_id_t          new_vid,
        const layer_id_t           target_level_id,
        const std::vector<nbr_t>&  pruned_results,
        const std::size_t          max_write_count
    ) -> void {
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
    }

    /**
     * @brief For each neighbor in @p pruned_results, append @p new_vid
     *        to that neighbor's own slot at @p target_level_id.
     *
     * If the neighbor's slot is not yet full, the new back-edge is
     * appended (with a trailing invalid sentinel if space remains).
     * If the slot is full, existing entries plus the new back-edge are
     * re-pruned via @p pruning_updater and rewritten.
     */
    template <typename PruningUpdaterT>
    __attribute__((always_inline))
    static auto _write_reverse_edges(
        this_index_t&              index,
        const vertex_id_t          new_vid,
        const layer_id_t           target_level_id,
        const std::vector<nbr_t>&  pruned_results,
        PruningUpdaterT&           pruning_updater
    ) -> void {
        for (std::size_t i = 0; i < pruned_results.size(); ++i) {
            const vertex_id_t nbr_vid  = pruned_results[i].get_vid();
            const distance_t  nbr_dist = pruned_results[i].get_distance();
            const nbr_t new_reverse_nbr = nbr_t::make_new_nbr(new_vid, nbr_dist);

            index.with_locked_nbrs(nbr_vid, target_level_id, [&](std::span<nbr_t> slot, vertex_num_t cnt) {
                const vertex_num_t slot_cap = static_cast<vertex_num_t>(slot.size());
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
                pruning_updater.update_impl(merged, slot_cap);
                for (std::size_t k = 0; k < merged.size() && k < slot_cap; ++k) {
                    slot[k] = merged[k];
                }
                if (merged.size() < slot_cap) {
                    slot[merged.size()] = nbr_t::make_invalid_nbr();
                }
            });
        }
    }

};  // class IndexFactory

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
