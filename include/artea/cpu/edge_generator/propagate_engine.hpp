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

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>
#include <artea/cpu/utils/bit_ops.hpp>

namespace artea {
namespace cpu {

template <typename ConstructorTraitsT>
class PropagateEngine {

    using vertex_num_t = typename ConstructorTraitsT::vertex_num_t;
    using vec_ele_t = typename ConstructorTraitsT::vec_ele_t;
    using vertex_id_t = typename ConstructorTraitsT::vertex_id_t;
    using distance_t = typename ConstructorTraitsT::distance_t;
    using iter_t = typename ConstructorTraitsT::iter_t;
    using nbr_t = typename ConstructorTraitsT::nbr_t;
    using nbr_arr_t = typename ConstructorTraitsT::nbr_arr_t;
    using log_buffer_t = typename ConstructorTraitsT::log_buffer_t;
    using log_container_t = typename ConstructorTraitsT::log_container_t;
    using log_table_t = typename ConstructorTraitsT::log_table_t;
    using index_graph_t = typename ConstructorTraitsT::index_graph_t;
    using word_aligned_bitmap_t = typename ConstructorTraitsT::word_aligned_bitmap_t;

    static constexpr bool selective_schedule = ConstructorTraitsT::selective_schedule;

public:
    PropagateEngine(index_graph_t& index_graph) :
        _log_table(index_graph.get_num_vertices()),
        _index_graph(index_graph),
        _executor_bitmap(index_graph.get_num_vertices())
    {}

    template <typename udf_updater_t>
    auto propagate(
        const vertex_id_t pivot_vid,
        udf_updater_t& udf_updater
    ) -> void {
        #ifndef NDEBUG
        if constexpr (selective_schedule) {
            if (!_executor_bitmap.test(pivot_vid)) {
                // No pending operations; skip propagation
                return;
            }
        }
        #endif

        nbr_arr_t& origin_nbrs = _index_graph.fetch_nbrs(pivot_vid);
        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.size());
        udf_updater(pivot_vid, origin_nbrs, retained_nbrs);
        std::swap(origin_nbrs, retained_nbrs);
    }

    template <typename udf_updater_t>
    auto propagate(udf_updater_t& udf_updater) -> void {
        const vertex_num_t num_vertices = _index_graph.get_num_vertices();

        // Dense Mode: Iterate all vertices
        if constexpr (not selective_schedule) {
            tbb::parallel_for(
                tbb::blocked_range<vertex_id_t>(0, num_vertices),
                [&](const tbb::blocked_range<vertex_id_t>& r) {
                    for (vertex_id_t pivot_vid = r.begin(); pivot_vid != r.end(); ++pivot_vid) {
                        propagate<udf_updater_t>(pivot_vid, udf_updater);
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
                            // Find index of the least significant bit (0-63)
                            int offset = count_trailing_zeros(word_mask);

                            vertex_id_t pivot_vid = base_vid + offset;

                            // Boundary check for the very last word
                            if (pivot_vid < num_vertices) {
                                propagate<udf_updater_t>(pivot_vid, udf_updater);
                            }

                            // Clear the bit we just processed
                            word_mask &= (word_mask - 1);
                        }
                    }
                }
            );  // end tbb::parallel_for
        }
    }

    /** @brief Merge the logged operations for a single vertex back to the index graph.
      * @param executor_vid The vertex id whose logged operations are to be merged.
    */
    __attribute__((always_inline))
    auto merge_logs(const vertex_id_t executor_vid) -> void {
        // Logic decoupled to NbrLogTable
        _log_table.apply_logs(executor_vid, _index_graph);
    }

    /** @brief Merge the logged operations for all vertices back to the index graph. */
    auto merge_logs() -> void {
        // Dense Mode: Iterate all vertices
        if constexpr (not selective_schedule) {
            const vertex_num_t num_vertices = _index_graph.get_num_vertices();
            tbb::parallel_for(
                tbb::blocked_range<vertex_id_t>(0, num_vertices),
                [&](const tbb::blocked_range<vertex_id_t>& r) {
                    for (vertex_id_t executor_vid = r.begin(); executor_vid != r.end(); ++executor_vid) {
                        merge_logs(executor_vid);
                    }
                }
            );  // end tbb::parallel_for
        }
        // Sparse Mode: Word Skipping + Bit Scanning
        else {
            _executor_bitmap.clear();

            const size_t num_words = _executor_bitmap.get_num_words();

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
                    for (size_t word_idx = r.begin(); word_idx != r.end(); ++word_idx) {
                        uint64_t next_round_mask = 0;
                        size_t start_vid, end_vid;
                        _executor_bitmap.get_range_from_word(word_idx, start_vid, end_vid);
                        // Check logs for each vertex in this word chunk
                        for (vertex_id_t vid = start_vid; vid < end_vid; ++vid) {
                            auto& log_container = _log_table.get_log_container(vid);
                            if (!log_container.empty()) {
                                _log_table.apply_logs(vid, _index_graph);
                                next_round_mask |= (1ULL << (vid & word_aligned_bitmap_t::WORD_MASK));
                            }
                        }
                        if (next_round_mask != 0) {
                            _executor_bitmap.set_word_mask(word_idx, next_round_mask);
                        }
                    }
                }
            );
        }
    }

    template <typename udf_updater_t, bool selective_schedule>
    auto next(udf_updater_t& udf_updater) -> void {
        propagate<udf_updater_t>(udf_updater);
        merge_logs();
    }

    template <typename udf_updater_t>
    auto run(const iter_t num_iters, udf_updater_t& udf_updater) -> void {
        for (iter_t iter = 0; iter < num_iters; ++iter) {
            next<udf_updater_t>(udf_updater);
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

private:

    /** @brief Operation log table for recording graph operations during propagation. */
    log_table_t _log_table;

    /** @brief Reference to the index graph being propagated. */
    index_graph_t& _index_graph;

    /** @brief Bitmap to track which vertices have pending operations. */
    word_aligned_bitmap_t _executor_bitmap;

};  // class PropagateEngine

}   // namespace cpu
}   // namespace artea