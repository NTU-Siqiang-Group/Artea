// --- START OF FILE nbr_log_table.hpp ---

// Copyright 2025 Weitang Ye
// ... (License omitted) ...

#pragma once

#include <cstddef>
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>

#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/utils/nbr_arr_checker.hpp>
#include <artea/common/definitions.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/propagation/graph_op_log.hpp>

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

public:

    NbrLogTable(const vertex_num_t num_vertices) {
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
    auto add_append_log(
        const vertex_id_t executor_vid,
        const vertex_id_t nbr_id,
        const distance_t new_edge_dist
    ) -> void {
        #ifndef NDEBUG
        if (is_nan_distance(new_edge_dist)) {
            logger.error("Attempted to log an operation with NaN distance.");
            throw std::runtime_error("Error: Logging an operation with NaN distance is not allowed.");
        }
        #endif
        // is_new = true, is_removed = false
        _nbr_logs[executor_vid].append(nbr_id, new_edge_dist, true, false);
    }

    /**
     * @brief Append a neighbor removal log.
     * @param executor_vid The vertex executing the operation.
     * @param nbr_id The neighbor vertex to be removed/sacrificed.
     * @param removed_edge_dist The distance of the edge to be removed.
     * @param direction The direction of the operation (IN/OUT).
     */
    __attribute__((always_inline))
    auto add_remove_log(
        const vertex_id_t executor_vid,
        const vertex_id_t nbr_id,
        const distance_t removed_edge_dist
    ) -> void {
        _nbr_logs[executor_vid].append(nbr_id, removed_edge_dist, false, true);
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
     *        Precondition: graph neighbors are sorted by StrictNeighborComparator (Dist, ID, removed).
     */
    template <typename GraphType>
    auto apply_logs(const vertex_id_t executor_vid, GraphType& graph) -> void {
        auto& log_container = _nbr_logs[executor_vid].get_container();
        if (log_container.empty()) return;

        auto& cur_nbrs = fetch_nbrs(executor_vid);

        #ifndef NDEBUG
        if (!NbrArrChecker<vertex_id_t, distance_t>::full_check(cur_nbrs)) {
            logger.error("Current neighbor array failed integrity check before applying logs.");
            throw std::runtime_error("Error: Current neighbor array failed integrity check before applying logs.");
        }
        #endif

        // Sort logs by StrictNeighborComparator (Distance first, then ID, then removed tag).
        std::sort(log_container.begin(), log_container.end(), StrictNeighborComparator<vertex_id_t, distance_t>);

        // Merge Logs (Sorted) into Graph (Sorted)
        auto middle_iter = cur_nbrs.insert(cur_nbrs.end(), log_container.begin(), log_container.end());
        std::inplace_merge(cur_nbrs.begin(), middle_iter, cur_nbrs.end(), StrictNeighborComparator<vertex_id_t, distance_t>);

        // Step 3: Duplicate (Linear Scanning)
        std::size_t stack_top = 0;

        for (std::size_t read_idx = 0; read_idx < cur_nbrs.size(); ++read_idx) {
            const auto& item = cur_nbrs[read_idx];
            bool match_prev = false;
            if (stack_top > 0) {
                const auto& prev = cur_nbrs[stack_top - 1];
                if (prev.get_id() == item.get_id()) match_prev = true;
            }
            if (item.is_removed()) {
                if (match_prev) stack_top--;    // pop stack
            }
            else {
                if (match_prev) continue;       // skip duplicate
                if (stack_top != read_idx) cur_nbrs[stack_top] = item;
                stack_top++;
            }
        }

        cur_nbrs.resize(stack_top);
        log_container.clear();

        #ifndef NDEBUG
        if (!NbrArrChecker<vertex_id_t, distance_t>::full_check(cur_nbrs)) {
            logger.error("Current neighbor array failed integrity check after applying logs.");
            throw std::runtime_error("Error: Current neighbor array failed integrity check after applying logs.");
        }
        #endif
    }

private:

    /** @brief Neighbor buffers for each vertex to store out-neighbor update logs. */
    cache_aligned_container_t<log_buffer_t> _nbr_logs;
};

}   // namespace cpu
}   // namespace artea