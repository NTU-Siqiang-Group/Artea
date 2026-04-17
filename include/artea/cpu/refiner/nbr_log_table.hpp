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
 * @FilePath: /Artea/include/artea/cpu/refiner/nbr_log_table.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Neighbor log table for buffering neighbor updates before applying to the graph.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>

namespace artea {
namespace cpu {

template <typename BufferTraitsT>
class NbrLogTable {

    using vertex_num_t = typename BufferTraitsT::vertex_num_t;
    using vertex_id_t = typename BufferTraitsT::vertex_id_t;
    using vec_ele_t = typename BufferTraitsT::vec_ele_t;
    using distance_t = typename BufferTraitsT::distance_t;
    using nbr_t = typename BufferTraitsT::nbr_t;
    using nbr_arr_t = typename BufferTraitsT::nbr_arr_t;
    using log_buffer_t = typename BufferTraitsT::log_buffer_t;
    using log_container_t = typename BufferTraitsT::log_container_t;
    using nbr_arr_checker_t = typename BufferTraitsT::nbr_arr_checker_t;
    using strict_nbr_comp_t = typename BufferTraitsT::strict_nbr_comp_t;

    constexpr static strict_nbr_comp_t strict_nbr_comp {};

public:

    /**
     * @brief Default-constructed; caller must invoke @c resize before use.
     *        The "executor_vid" across this class is a @b local vid (row
     *        index in the RefiningGraph's _nbrs_arr). In identity-mapped
     *        RefiningGraphs this is bit-identical to the global vid; for a
     *        sparse layer RefiningGraph it is the row index within that
     *        layer only.
     */
    NbrLogTable() = default;

    /** @brief Allocate one log buffer per vertex in the RefiningGraph.
     *         For a sparse upper-layer RG @p num_vertices is the layer
     *         size (N_local), not the global vertex count. */
    auto resize(const vertex_num_t num_vertices) -> void {
        _nbr_logs.clear();
        _nbr_logs.resize(num_vertices);
    }

    /**
     * @brief Append a neighbor addition log.
     * @param executor_vid The vertex executing the operation.
     * @param nbr_id The neighbor vertex to be added/delegated.
     * @param new_edge_dist The distance of the new edge to be added.
     * @param direction The direction of the operation (IN/OUT).
     */
    __attribute__((always_inline))
    auto write_log(
        const vertex_id_t executor_vid,
        const vertex_id_t nbr_id,
        const distance_t new_edge_dist
    ) -> void {
        #ifndef NDEBUG
        if (BufferTraitsT::is_nan_distance(new_edge_dist)) {
            ARTEA_ERROR("Attempted to log an operation with NaN distance.");
        }
        #endif
        _nbr_logs[executor_vid].append(nbr_id, new_edge_dist, true); // is_new = true
    }

    /** @brief Batch-append multiple neighbor logs for a single executor vertex.
     *  @param executor_vid   The vertex executing the operation.
     *  @param nbr_ids        Container of neighbor vertex IDs.
     *  @param new_edge_dists Container of corresponding distances (same length).
     */
    template <typename NbrIdContainerT, typename DistContainerT>
    __attribute__((always_inline))
    auto write_logs(
        const vertex_id_t executor_vid,
        const NbrIdContainerT& nbr_ids,
        const DistContainerT& new_edge_dists
    ) -> void {
        auto& container = get_log_container(executor_vid);
        const size_t count = nbr_ids.size();
        container.reserve(container.size() + count);
        for (size_t i = 0; i < count; ++i) {
            container.emplace_back(nbr_ids[i], new_edge_dists[i], true); // is_new = true
        }
    }

    __attribute__((always_inline))
    auto clear_logs(const vertex_id_t executor_vid) -> void {
        _nbr_logs[executor_vid].clear();
    }

    __attribute__((always_inline))
    auto get_log_container(const vertex_id_t executor_vid) -> log_container_t& {
        return _nbr_logs[executor_vid].get_container();
    }

    /**
     * @brief Applies logs to the graph directly using Distance ordering.
     *        Precondition: graph neighbors are sorted by StrictNbrComparator (Dist, ID).
     * @param executor_vid The vertex whose logs are to be applied.
     * @param graph The bottom layer graph to which the logs will be applied.
     * @return Number of logs applied.
     * @note This function can be called thread-safely for different executor_vids in parallel.
     */
    template <typename GraphType>
    auto apply_logs(const vertex_id_t executor_vid, GraphType& graph) -> size_t {
        auto& log_container = _nbr_logs[executor_vid].get_container();
        if (log_container.empty()) return 0;

        const size_t num_logs = log_container.size();

        // executor_vid is local; translate to global for the RG row fetch.
        auto& cur_nbrs = graph.fetch_nbrs(graph.get_storage_vid(executor_vid));

        #ifndef NDEBUG
        if (!nbr_arr_checker_t::full_check(cur_nbrs)) {
            ARTEA_ERROR("Error: Integrity check failed before applying logs.");
        }
        #endif

        // * Sort the logs (Update buffer)
        // * Complexity: O(M log M), where M is number of logs (usually small).
        std::sort(log_container.begin(), log_container.end(), strict_nbr_comp);

        // * Insert and Merge

        // * Expand vector to hold everything
        auto old_size = cur_nbrs.size();
        cur_nbrs.reserve(old_size + log_container.size());

        // * Append logs to the end. O(1)
        auto middle_iter = cur_nbrs.insert(cur_nbrs.end(), log_container.begin(), log_container.end());

        // * Clear the log buffer early
        log_container.clear();

        // * Merge the two sorted ranges: [begin, middle) and [middle, end). O(N + M)
        std::inplace_merge(cur_nbrs.begin(), middle_iter, cur_nbrs.end(), strict_nbr_comp);

        // * Deduplicate (std::unique)
        // * Note that neighbors with same ID must have same distance (guaranteed by the nature of bottom layer graph)
        auto last = std::unique(cur_nbrs.begin(), cur_nbrs.end(),
            [](const auto& a, const auto& b) {
                return a.get_vid() == b.get_vid();
            });

        // * Erase the undefined elements at the tail
        cur_nbrs.erase(last, cur_nbrs.end());

        #ifndef NDEBUG
        if (!nbr_arr_checker_t::full_check(cur_nbrs)) {
            ARTEA_ERROR(fmt::format(
                "Neighbor array integrity check failed after applying logs for vertex {}.",
                executor_vid
            ));
        }
        #endif

        return num_logs;
    }

private:

    /** @brief Neighbor buffers for each vertex to store out-neighbor update logs. */
    cache_aligned_container_t<log_buffer_t> _nbr_logs;
};

}   // namespace cpu
}   // namespace artea