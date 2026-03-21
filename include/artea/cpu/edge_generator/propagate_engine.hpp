// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/propagate_engine.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Propagation engine for edge generation.
 */

#pragma once

#include <vector>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <bit>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

namespace artea {
namespace cpu {

template <typename EdgeGeneratorTraitsT, bool SelectiveSchedule = false>
class PropagateEngine {

    using vertex_num_t = typename EdgeGeneratorTraitsT::vertex_num_t;
    using vec_ele_t = typename EdgeGeneratorTraitsT::vec_ele_t;
    using vertex_id_t = typename EdgeGeneratorTraitsT::vertex_id_t;
    using distance_t = typename EdgeGeneratorTraitsT::distance_t;
    using iter_t = typename EdgeGeneratorTraitsT::iter_t;
    using nbr_t = typename EdgeGeneratorTraitsT::nbr_t;
    using nbr_arr_t = typename EdgeGeneratorTraitsT::nbr_arr_t;
    using log_buffer_t = typename EdgeGeneratorTraitsT::log_buffer_t;
    using log_container_t = typename EdgeGeneratorTraitsT::log_container_t;
    using log_table_t = typename EdgeGeneratorTraitsT::log_table_t;
    using word_aligned_bitmap_t = typename EdgeGeneratorTraitsT::word_aligned_bitmap_t;
    using flat_graph_t = typename EdgeGeneratorTraitsT::flat_graph_t;
    using dist_func_t = typename EdgeGeneratorTraitsT::dist_func_t;

    template <typename DerivedClassT>
    using neighbor_updater_t = typename EdgeGeneratorTraitsT::template neighbor_updater_t<DerivedClassT>;

    static constexpr bool selective_schedule = SelectiveSchedule;
    static constexpr bool profiling_mode = EdgeGeneratorTraitsT::profiling_mode;

public:
    PropagateEngine(const vertex_num_t num_vertices, const dist_func_t& dist_func) :
        _log_table(num_vertices),
        _executor_bitmap(num_vertices),
        _flat_graph(nullptr),
        _dist_func(dist_func)
    {}

    /** @brief Set the flat graph to operate on. */
    __attribute__((always_inline))
    auto set_graph(flat_graph_t& flat_graph) -> void {
        _flat_graph = &flat_graph;

        // Initialize executor bitmap for selective scheduling
        if constexpr (selective_schedule) {
            _executor_bitmap.set_all();
        }
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto propagate(
        const vertex_id_t pivot_vid,
        UdfUpdaterT& udf_updater
    ) -> void {
        nbr_arr_t& origin_nbrs = _flat_graph->fetch_nbrs(pivot_vid);
        udf_updater(pivot_vid, origin_nbrs);
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto propagate(UdfUpdaterT& udf_updater) -> void {
        const vertex_num_t num_vertices = _flat_graph->get_num_vertices();

        // Dense Mode: Iterate all vertices
        if constexpr (not selective_schedule) {
            tbb::parallel_for(
                tbb::blocked_range<vertex_id_t>(0, num_vertices),
                [&](const tbb::blocked_range<vertex_id_t>& r) {
                    for (vertex_id_t pivot_vid = r.begin(); pivot_vid != r.end(); ++pivot_vid) {
                        propagate<UdfUpdaterT>(pivot_vid, udf_updater);
                    }
                }
            );
        }
        // Sparse Mode: Word Skipping + Bit Scanning
        else {
            const size_t num_words = _executor_bitmap.get_num_words();

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_words),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t word_idx = r.begin(); word_idx != r.end(); ++word_idx) {
                        uint64_t word_mask = _executor_bitmap.get_word_mask(word_idx);
                        // Fast skip: If no bits are set, skip 64 vertices instantly
                        if (word_mask == 0) continue;
                        const vertex_id_t base_vid = word_idx << word_aligned_bitmap_t::WORD_SHIFT;
                        // Bit Scanning: Iterate only set bits
                        while (word_mask != 0) {
                            // Fast Locating: Find index of the least significant bit (0-63)
                            int offset = std::countr_zero(word_mask);

                            vertex_id_t pivot_vid = base_vid + offset;

                            // Boundary check for the very last word (rarely false)
                            if (__builtin_expect(pivot_vid < num_vertices, 1)) {
                                propagate<UdfUpdaterT>(pivot_vid, udf_updater);
                            }

                            // Clear the bit we just processed
                            word_mask &= (word_mask - 1);
                        }
                    }
                }
            );  // end tbb::parallel_for
        }
    }

    /** @brief Merge the logged operations for a single vertex back to the flat graph.
      * @param executor_vid The vertex id whose logged operations are to be merged.
      * @return Number of logs merged.
    */
    __attribute__((always_inline))
    auto merge_logs(const vertex_id_t executor_vid) -> size_t {
        // Logic decoupled to NbrLogTable
        return _log_table.apply_logs(executor_vid, *_flat_graph);
    }

    /** @brief Merge the logged operations for all vertices back to the flat graph.
     *  @return Total number of logs merged in this call.
     */
    auto merge_logs() -> size_t {
        // Dense Mode: Iterate all vertices
        if constexpr (not selective_schedule) {
            const vertex_num_t num_vertices = _flat_graph->get_num_vertices();

            // Thread-local counters for lock-free statistics
            tbb::enumerable_thread_specific<size_t> local_counts;

            tbb::parallel_for(
                tbb::blocked_range<vertex_id_t>(0, num_vertices),
                [&](const tbb::blocked_range<vertex_id_t>& r) {
                    size_t& local_count = local_counts.local();
                    for (vertex_id_t executor_vid = r.begin(); executor_vid != r.end(); ++executor_vid) {
                        local_count += merge_logs(executor_vid);
                    }
                }
            );  // end tbb::parallel_for

            if constexpr (profiling_mode) {
                _merged_logs_count = local_counts.combine(std::plus<size_t>());
            }
        }
        // Sparse Mode: Word Skipping + Bit Scanning
        else {
            _executor_bitmap.clear();

            const size_t num_words = _executor_bitmap.get_num_words();

            // Thread-local counters for lock-free statistics
            tbb::enumerable_thread_specific<size_t> local_counts;

            #ifndef NDEBUG
            if (num_words != _executor_bitmap.get_num_words()) {
                logger.error("Inconsistent executor bitmap word counts between IN and OUT.");
                throw std::runtime_error("Error: Inconsistent executor bitmap word counts between IN and OUT.");
            }
            #endif

            // Parallel Dimension: Word Index (0 ... N/64)
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_words),
                [&](const tbb::blocked_range<size_t>& r) {
                    size_t& local_count = local_counts.local();
                    for (size_t word_idx = r.begin(); word_idx != r.end(); ++word_idx) {
                        uint64_t next_round_mask = 0;
                        size_t start_vid, end_vid;
                        _executor_bitmap.get_range_from_word(word_idx, start_vid, end_vid);
                        // Check logs for each vertex in this word chunk
                        for (vertex_id_t vid = start_vid; vid < end_vid; ++vid) {
                            auto& log_container = _log_table.get_log_container(vid);
                            if (!log_container.empty()) {
                                local_count += merge_logs(vid);
                                next_round_mask |= (1ULL << (vid & word_aligned_bitmap_t::WORD_MASK));
                            }
                        }
                        if (next_round_mask != 0) {
                            _executor_bitmap.set_word_mask(word_idx, next_round_mask);
                        }
                    }
                }
            );

            if constexpr (profiling_mode) {
                _merged_logs_count = local_counts.combine(std::plus<size_t>());
            }
        }

        if constexpr (profiling_mode) {
            return _merged_logs_count;
        } else {
            return 0;
        }
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto next(UdfUpdaterT& udf_updater) -> void {
        propagate<UdfUpdaterT>(udf_updater);
        merge_logs();
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto run(const iter_t num_iters, UdfUpdaterT& udf_updater) -> void {
        for (iter_t iter = 0; iter < num_iters; ++iter) {
            next<UdfUpdaterT>(udf_updater);
            if constexpr (profiling_mode) {
                logger.info(fmt::format(
                    "Inner Iter {} ({}): Merged {} logs",
                    iter,
                    UdfUpdaterT::updater_name,
                    _merged_logs_count
                ));
            }
        }
    }

    /** @brief Accessor for the internal DelegateTable. */
    __attribute__((always_inline))
    auto get_log_table() -> log_table_t& {
        return _log_table;
    }

    __attribute__((always_inline))
    auto get_executor_bitmap() -> word_aligned_bitmap_t& {
        return _executor_bitmap;
    }

    /** @brief Get the total count of merged logs across all merge_logs() calls.
     *  @return Total number of logs merged.
     */
    __attribute__((always_inline))
    auto get_merged_logs_count() const -> size_t {
        if constexpr (profiling_mode) {
            return _merged_logs_count;
        } else {
            return 0;
        }
    }

    /** @brief Factory method to create an updater of the specified type.
      * @tparam UpdaterT The updater type to create (e.g., triangle_updater_t, reverse_updater_t, random_updater_t).
      * @tparam Args Variadic template for additional constructor arguments.
      * @param args Additional arguments specific to the updater type.
      * @return An instance of the requested updater type.
      *
      * @note This factory method automatically provides dist_func, vecs_arr, log_table,
      *       and max_nbr_size from the internal state. User only needs
      *       to provide updater-specific parameters.
      *
      * Supported updaters:
      *   - triangle_updater_t: Requires scale_coeffs and optional shifted_coeffs
      *   - reverse_updater_t: No additional parameters required
      *   - random_updater_t: Requires rand_gen_size (num_vertices is auto-provided)
      */
    template <typename UpdaterT, typename... Args>
    auto make_updater(Args&&... args) -> UpdaterT {
        using triangle_updater_t = typename EdgeGeneratorTraitsT::triangle_updater_t;
        using reverse_updater_t = typename EdgeGeneratorTraitsT::reverse_updater_t;
        using random_updater_t = typename EdgeGeneratorTraitsT::random_updater_t;

        const auto& vecs_arr = _flat_graph->get_vecs_data();
        auto& log_table = _log_table;
        const auto max_nbr_size = _flat_graph->layer_config().max_nbr_size();
        const auto num_vertices = _flat_graph->get_num_vertices();

        if constexpr (std::is_same_v<UpdaterT, triangle_updater_t>) {
            // TriangleUpdater constructor signature:
            // TriangleUpdater(dist_func, vecs_arr, log_table, max_nbr_size, scale_coeffs, shifted_coeffs)
            return UpdaterT(_dist_func, vecs_arr, log_table, max_nbr_size, std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<UpdaterT, reverse_updater_t>) {
            // ReverseUpdater constructor signature:
            // ReverseUpdater(dist_func, vecs_arr, log_table)
            return UpdaterT(_dist_func, vecs_arr, log_table);
        } else if constexpr (std::is_same_v<UpdaterT, random_updater_t>) {
            // RandomUpdater constructor signature:
            // RandomUpdater(dist_func, vecs_arr, log_table, num_vertices, rand_gen_size)
            return UpdaterT(_dist_func, vecs_arr, log_table, num_vertices, std::forward<Args>(args)...);
        } else {
            throw std::runtime_error("Unsupported updater type");
        }
    }

private:

    /** @brief Operation log table for recording graph operations during propagation. */
    log_table_t _log_table;

    /** @brief Bitmap to track which vertices have pending operations. */
    word_aligned_bitmap_t _executor_bitmap;

    /** @brief Pointer to the flat graph being operated on. */
    flat_graph_t* _flat_graph;

    /** @brief Distance function reference. */
    const dist_func_t& _dist_func;

    /** @brief Total count of merged logs (accumulated across all merge_logs() calls). */
    size_t _merged_logs_count = 0;

};  // class PropagateEngine

}   // namespace cpu
}   // namespace artea
