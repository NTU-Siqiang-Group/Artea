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
 * @FilePath: /Artea/include/artea/cpu/propagation/stacked_propagate.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <vector>
#include <utility>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/word_aligned_bitmap.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/index/neighbor.hpp>
// NbrArrChecker and ArraySearch moved/removed if not used directly here
#include <artea/cpu/propagation/graph_op_log.hpp>
#include <artea/cpu/propagation/nbr_log_table.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename log_buffer_t
>
class PropagateEngine {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;
    using log_container_t = typename log_buffer_t::container_t;

public:
    PropagateEngine(IndexGraph<vertex_num_t, vec_ele_t, graph_direction_t::HIBRID>& index_graph) :
        _log_table(index_graph.get_num_vertices()),
        _index_graph(index_graph),
        _in_executor_bitmap(index_graph.get_num_vertices()),
        _out_executor_bitmap(index_graph.get_num_vertices())
    {}

    template <typename udf_updater_t, bool selective_schedule, op_direction_t op_direction>
    auto propagate(
        const vertex_id_t pivot_vid,
        udf_updater_t& udf_updater
    ) -> void {
        if constexpr (selective_schedule) {
            auto& executor_bitmap = (op_direction == op_direction_t::IN) ?
                _in_executor_bitmap : _out_executor_bitmap;
            if (!executor_bitmap.test(pivot_vid)) {
                // No pending operations; skip propagation
                return;
            }
        }

        nbr_arr_t& origin_nbrs = _index_graph.template fetch_nbrs<op_direction>(pivot_vid);
        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.size());
        udf_updater(pivot_vid, op_direction, origin_nbrs, retained_nbrs);
        std::swap(origin_nbrs, retained_nbrs);
    }

    template <typename udf_updater_t, bool selective_schedule>
    auto propagate(udf_updater_t& udf_updater) -> void {
        const vertex_num_t num_vertices = _index_graph.get_num_vertices();
        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t pivot_vid = r.begin(); pivot_vid != r.end(); ++pivot_vid) {
                    propagate<udf_updater_t, selective_schedule, op_direction_t::OUT>(
                        pivot_vid, udf_updater);
                    propagate<udf_updater_t, selective_schedule, op_direction_t::IN>(
                        pivot_vid, udf_updater);
                }
            }
        );
    }

    /** @brief Merge the logged operations for a single vertex back to the index graph.
      * @param executor_vid The vertex id whose logged operations are to be merged.
    */
    __attribute__((always_inline))
    auto merge_logs(const vertex_id_t executor_vid) -> void {
        // Logic decoupled to NbrLogTable
        _log_table.apply_logs(executor_vid, _index_graph, op_direction_t::IN);
        _log_table.apply_logs(executor_vid, _index_graph, op_direction_t::OUT);
    }

    /** @brief Merge the logged operations for all vertices back to the index graph. */
    template <bool selective_schedule>
    auto merge_logs() -> void {
        if constexpr (not selective_schedule) {
            const vertex_num_t num_vertices = _index_graph.get_num_vertices();
            tbb::parallel_for(
                tbb::blocked_range<vertex_id_t>(0, num_vertices),
                [&](const tbb::blocked_range<vertex_id_t>& r) {
                    for (vertex_id_t executor_vid = r.begin(); executor_vid != r.end(); ++executor_vid) {
                        merge_logs(executor_vid);
                    }
                }
            );
        }
        // Selective Scheduling based on Bitmap
        else {
            const size_t num_words = _in_executor_bitmap.get_num_words();

            #ifndef NDEBUG
            if (num_words != _out_executor_bitmap.get_num_words()) {
                logger.error("Inconsistent executor bitmap word counts between IN and OUT.");
                throw std::runtime_error("Error: Inconsistent executor bitmap word counts between IN and OUT.");
            }
            #endif

            // Parallel Dimension: Word Index (0 ... N/64)
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_words),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t word_idx = r.begin(); word_idx != r.end(); ++word_idx) {
                        uint64_t current_in_mask = 0, current_out_mask = 0;
                        size_t start_vid, end_vid;
                        _in_executor_bitmap.get_range_from_word(word_idx, start_vid, end_vid);
                        for (vertex_id_t vid = start_vid; vid < end_vid; ++vid) {
                            auto& log_in = _log_table.get_log_container(vid, op_direction_t::IN);
                            if (!log_in.empty()) {
                                _log_table.apply_logs(vid, _index_graph, op_direction_t::IN);
                                current_in_mask |= (1ULL << (vid & WordAlignedBitmap::WORD_MASK));
                            }
                            auto& log_out = _log_table.get_log_container(vid, op_direction_t::OUT);
                            if (!log_out.empty()) {
                                _log_table.apply_logs(vid, _index_graph, op_direction_t::OUT);
                                current_out_mask |= (1ULL << (vid & WordAlignedBitmap::WORD_MASK));
                            }
                        }
                        if (current_in_mask != 0) {
                            _in_executor_bitmap.set_word_mask(word_idx, current_in_mask);
                        }
                        if (current_out_mask != 0) {
                            _out_executor_bitmap.set_word_mask(word_idx, current_out_mask);
                        }
                    }
                }
            );
        }
    }

    template <typename udf_updater_t, bool selective_schedule>
    auto next(udf_updater_t& udf_updater) -> void {
        propagate<udf_updater_t, selective_schedule>(udf_updater);
        merge_logs<selective_schedule>();
    }

    template <typename udf_updater_t, bool selective_schedule>
    auto run(const iter_t num_iters, udf_updater_t& udf_updater) -> void {
        for (iter_t iter = 0; iter < num_iters; ++iter) {
            next<udf_updater_t, selective_schedule>(udf_updater);
        }
    }

    /** @brief Accessor for the internal DelegateTable. */
    __attribute__((always_inline))
    auto get_log_table() -> NbrLogTable<vertex_num_t, vec_ele_t, log_buffer_t>& {
        return _log_table;
    }

    __attribute__((always_inline))
    auto get_in_executor_bitmap() -> WordAlignedBitmap& {
        return _in_executor_bitmap;
    }

    __attribute__((always_inline))
    auto get_out_executor_bitmap() -> WordAlignedBitmap& {
        return _out_executor_bitmap;
    }

private:

    /** @brief Operation log table for recording graph operations during propagation. */
    NbrLogTable<vertex_num_t, vec_ele_t, log_buffer_t> _log_table;

    /** @brief Reference to the index graph being propagated. */
    IndexGraph<vertex_num_t, vec_ele_t, graph_direction_t::HIBRID>& _index_graph;

    /** @brief Bitmap to track which vertices have pending operations. */
    WordAlignedBitmap _in_executor_bitmap;
    WordAlignedBitmap _out_executor_bitmap;

};  // class PropagateEngine

}   // namespace cpu
}   // namespace artea