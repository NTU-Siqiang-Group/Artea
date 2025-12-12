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
#include <artea/definitions.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/propagation/graph_op_log.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename nbr_buffer_t
>
class NbrLogTable {

    using vertex_id_t = uint32_t;
    using distance_t = float;
    using nbr_t = Neighbor<vertex_id_t, distance_t>;
    using nbr_container_t = typename nbr_buffer_t::container_t;

public:

    NbrLogTable(const vertex_num_t num_vertices) {
        _in_nbr_logs.resize(num_vertices);
        _out_nbr_logs.resize(num_vertices);
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
        const distance_t new_edge_dist,
        const op_direction_t direction
    ) -> void {
        #ifndef NDEBUG
        if (is_nan_distance(new_edge_dist)) {
            logger.error("Attempted to log an operation with NaN distance.");
            throw std::runtime_error("Error: Logging an operation with NaN distance is not allowed.");
        }
        #endif
        if (direction == op_direction_t::IN) {
            _in_nbr_logs[executor_vid].append(nbr_id, new_edge_dist, true, false);  // is_new = true, is_removed = false
        }
        else { // OUT direction
            _out_nbr_logs[executor_vid].append(nbr_id, new_edge_dist, true, false); // is_new = true, is_removed = false
        }
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
        const distance_t removed_edge_dist,
        const op_direction_t direction
    ) -> void {
        if (direction == op_direction_t::IN) {
            _in_nbr_logs[executor_vid].append(nbr_id, removed_edge_dist, false, true);  // is_new = false (irrelevant for remove), is_removed = true
        } else { // OUT direction
            _out_nbr_logs[executor_vid].append(nbr_id, removed_edge_dist, false, true); // is_new = false (irrelevant for remove), is_removed = true
        }
    }

    __attribute__((always_inline))
    auto clear_logs(const vertex_id_t executor_vid, const op_direction_t direction) -> void {
        if (direction == op_direction_t::IN) {
            _in_nbr_logs[executor_vid].clear();
        } else {
            _out_nbr_logs[executor_vid].clear();
        }
    }

    __attribute__((always_inline))
    auto clear_logs(const vertex_id_t executor_vid) -> void {
        _in_nbr_logs[executor_vid].clear();
        _out_nbr_logs[executor_vid].clear();
    }

    __attribute__((always_inline))
    auto get_log_container(const vertex_id_t executor_vid, const op_direction_t direction) -> nbr_container_t& {
        return direction == op_direction_t::IN ?
            _in_nbr_logs[executor_vid].get_container() : _out_nbr_logs[executor_vid].get_container();
    }

    /**
     * @brief Applies logs to the graph directly using Distance ordering.
     *        Precondition: graph neighbors are sorted by StrictNeighborComparator (Dist, ID, removed).
     */
    template <typename GraphType>
    auto apply_logs(const vertex_id_t executor_vid, GraphType& graph, const op_direction_t direction) -> void {

        auto& log_container = direction == op_direction_t::IN ?
            _in_nbr_logs[executor_vid].get_container() :
            _out_nbr_logs[executor_vid].get_container();

        if (log_container.empty()) return;

        auto& cur_nbrs = direction == op_direction_t::IN ?
            graph.template fetch_nbrs<op_direction_t::IN>(executor_vid) :
            graph.template fetch_nbrs<op_direction_t::OUT>(executor_vid);

        #ifndef NDEBUG
        if (!NbrArrChecker<vertex_id_t, distance_t>::no_nan_check(cur_nbrs)) {
            logger.error("Neighbor array contains NaN distances before applying logs.");
            throw std::runtime_error("Error: Neighbor array contains NaN distances before applying logs.");
        }
        if (!NbrArrChecker<vertex_id_t, distance_t>::no_removed_check(cur_nbrs)) {
            logger.error("Neighbor array contains removed neighbors before applying logs.");
            throw std::runtime_error("Error: Neighbor array contains removed neighbors before applying logs.");
        }
        if (!NbrArrChecker<vertex_id_t, distance_t>::no_duplicate_check(cur_nbrs)) {
            logger.error("Neighbor array contains duplicate neighbors before applying logs.");
            throw std::runtime_error("Error: Neighbor array contains duplicate neighbors before applying logs.");
        }
        if (!NbrArrChecker<vertex_id_t, distance_t>::distance_order_check(cur_nbrs)) {
            logger.error("Neighbor array is not sorted by distance before applying logs.");
            throw std::runtime_error("Error: Neighbor array is not sorted by distance before applying logs.");
        }
        #endif

        // Sort logs by StrictNeighborComparator (Distance first, then ID, then removed tag).
        std::sort(log_container.begin(), log_container.end(), StrictNeighborComparator<vertex_id_t, distance_t>);

        // Merge Logs (Sorted) into Graph (Sorted)
        auto middle_iter = cur_nbrs.insert(cur_nbrs.end(), log_container.begin(), log_container.end());
        std::inplace_merge(cur_nbrs.begin(), middle_iter, cur_nbrs.end(), StrictNeighborComparator<vertex_id_t, distance_t>);

        // Step 3: Linear Scan with Cancellation (Stack Logic)
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
        if (!NbrArrChecker<vertex_id_t, distance_t>::no_nan_check(cur_nbrs)) {
            logger.error("Neighbor array contains NaN distances before applying logs.");
            throw std::runtime_error("Error: Neighbor array contains NaN distances before applying logs.");
        }
        if (!NbrArrChecker<vertex_id_t, distance_t>::no_removed_check(cur_nbrs)) {
            logger.error("Neighbor array contains removed neighbors before applying logs.");
            throw std::runtime_error("Error: Neighbor array contains removed neighbors before applying logs.");
        }
        if (!NbrArrChecker<vertex_id_t, distance_t>::no_duplicate_check(cur_nbrs)) {
            logger.error("Neighbor array contains duplicate neighbors before applying logs.");
            throw std::runtime_error("Error: Neighbor array contains duplicate neighbors before applying logs.");
        }
        if (!NbrArrChecker<vertex_id_t, distance_t>::distance_order_check(cur_nbrs)) {
            logger.error("Neighbor array is not sorted by distance before applying logs.");
            throw std::runtime_error("Error: Neighbor array is not sorted by distance before applying logs.");
        }
        #endif
    }

private:

    /** @brief Neighbor buffers for each vertex to store in-neighbor update logs. */
    cache_aligned_container_t<nbr_buffer_t> _in_nbr_logs;

    /** @brief Neighbor buffers for each vertex to store out-neighbor update logs. */
    cache_aligned_container_t<nbr_buffer_t> _out_nbr_logs;
};

}   // namespace cpu
}   // namespace artea