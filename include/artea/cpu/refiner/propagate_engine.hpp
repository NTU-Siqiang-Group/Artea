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
 * @FilePath: /Artea/include/artea/cpu/refiner/propagate_engine.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Propagation engine for edge generation.
 */

#pragma once

#include <vector>
#include <utility>
#include <type_traits>
#include <stdexcept>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename RefinerTraitsT>
class PropagateEngine {

    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using iter_t = typename RefinerTraitsT::iter_t;
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using log_buffer_t = typename RefinerTraitsT::log_buffer_t;
    using log_container_t = typename RefinerTraitsT::log_container_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;

    template <typename DerivedClassT>
    using neighbor_updater_t = typename RefinerTraitsT::template neighbor_updater_t<DerivedClassT>;

    static constexpr bool profiling_mode = RefinerTraitsT::profiling_mode;

public:
    /**
     * @brief Iterate every participating vertex of @p refining_graph in
     *        parallel. The callback receives @c (local_vid, global_vid).
     *        In identity mode both are equal; in sparse mode
     *        @c local_vid is the row index in the RG and @c global_vid
     *        is the resolved global id via @c refining_graph.vid_at.
     *
     * Templated on the RefiningGraph type so that HierarchicalGraph
     * callers (which don't see RefinerTraits) can reuse the helper
     * without coupling.
     */
    template <typename RefiningGraphT, typename CallbackT>
    static auto parallel_for_each_vertex(
        const RefiningGraphT& refining_graph,
        CallbackT&&           callback
    ) -> void {
        using vn_t = decltype(refining_graph.get_num_vertices());
        tbb::parallel_for(
            tbb::blocked_range<vn_t>(0, refining_graph.get_num_vertices()),
            [&](const tbb::blocked_range<vn_t>& range) {
                for (vn_t i = range.begin(); i != range.end(); ++i) {
                    callback(i, refining_graph.vid_at(i));
                }
            });
    }

    /**
     * @brief Same as @c parallel_for_each_vertex but restricted to local
     *        row indices in @c [local_start, local_end). Used by ranged
     *        propagate overloads that only want to touch a sub-range of
     *        vertices (e.g. incremental batch seeding).
     */
    template <typename RefiningGraphT, typename CallbackT>
    static auto parallel_for_each_vertex_in_range(
        const RefiningGraphT& refining_graph,
        const vertex_num_t    local_start,
        const vertex_num_t    local_end,
        CallbackT&&           callback
    ) -> void {
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(local_start, local_end),
            [&](const tbb::blocked_range<vertex_num_t>& range) {
                for (vertex_num_t i = range.begin(); i != range.end(); ++i) {
                    callback(i, refining_graph.vid_at(i));
                }
            });
    }

    /**
     * @brief Construct with only the distance function. @c _log_table is
     *        default-constructed (empty) and sized on every @c set_graph
     *        call to match the bound RefiningGraph's vertex count (N_local
     *        for sparse layers, N_global for identity-mapped RGs).
     */
    PropagateEngine(const dist_func_t& dist_func) :
        _refining_graph(nullptr),
        _dist_func(dist_func)
    {}

    /** @brief Set the descent graph to operate on. */
    __attribute__((always_inline))
    auto set_graph(refining_graph_t& refining_graph) -> void {
        _refining_graph = &refining_graph;
        _log_table.resize(refining_graph.get_num_vertices());
    }

    /**
     * @brief Run the updater on a single pivot. @p local_vid indexes the
     *        log_table / RG row; the global vid is resolved via vid_at and
     *        handed to the updater alongside so it can touch _vecs_data
     *        (global-indexed) and write logs (local-indexed).
     */
    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto propagate(
        const vertex_id_t local_vid,
        const vertex_id_t global_vid,
        UdfUpdaterT& udf_updater
    ) -> void {
        nbr_arr_t& origin_nbrs = _refining_graph->fetch_nbrs(global_vid);
        udf_updater(local_vid, global_vid, origin_nbrs);
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto propagate(UdfUpdaterT& udf_updater) -> void {
        parallel_for_each_vertex(
            *_refining_graph,
            [&](const vertex_id_t local_vid, const vertex_id_t global_vid) {
                propagate<UdfUpdaterT>(local_vid, global_vid, udf_updater);
            });
    }

    /**
     * @brief Ranged propagate: runs @p udf_updater only on local row ids
     *        in @c [local_start, local_end). Used for incremental seeding
     *        where the outer loop must be restricted to a vid window.
     */
    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto propagate_range(
        UdfUpdaterT&       udf_updater,
        const vertex_num_t local_start,
        const vertex_num_t local_end
    ) -> void {
        parallel_for_each_vertex_in_range(
            *_refining_graph, local_start, local_end,
            [&](const vertex_id_t local_vid, const vertex_id_t global_vid) {
                propagate<UdfUpdaterT>(local_vid, global_vid, udf_updater);
            });
    }

    /** @brief Merge the logged operations for a single vertex back to the descent graph.
      * @param local_vid Local row index whose logged operations are to be merged.
      * @return Number of logs merged.
    */
    __attribute__((always_inline))
    auto merge_logs(const vertex_id_t local_vid) -> size_t {
        return _log_table.apply_logs(local_vid, *_refining_graph);
    }

    /** @brief Merge the logged operations for all vertices back to the descent graph.
     *  @return Total number of logs merged in this call.
     */
    auto merge_logs() -> size_t {
        tbb::enumerable_thread_specific<size_t> local_counts;

        parallel_for_each_vertex(
            *_refining_graph,
            [&](const vertex_id_t local_vid, const vertex_id_t /*global_vid*/) {
                local_counts.local() += merge_logs(local_vid);
            });

        if constexpr (profiling_mode) {
            _merged_logs_count = local_counts.combine(std::plus<size_t>());
            return _merged_logs_count;
        } else {
            return 0;
        }
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto next(UdfUpdaterT& udf_updater) -> PropagateEngine& {
        propagate<UdfUpdaterT>(udf_updater);
        merge_logs();
        return *this;
    }

    /**
     * @brief Ranged @c next: run @p udf_updater only on local row ids in
     *        @c [local_start, local_end), then merge logs for all rows.
     *        We still merge the full log_table because the updater's
     *        writes may have landed on rows outside the iteration window
     *        (e.g. reverse edges targeting other vids).
     */
    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto next_range(
        UdfUpdaterT&       udf_updater,
        const vertex_num_t local_start,
        const vertex_num_t local_end
    ) -> PropagateEngine& {
        propagate_range<UdfUpdaterT>(udf_updater, local_start, local_end);
        merge_logs();
        return *this;
    }

    template <typename UdfUpdaterT>
        requires std::derived_from<UdfUpdaterT, neighbor_updater_t<UdfUpdaterT>>
    auto run(const iter_t num_iters, UdfUpdaterT& udf_updater) -> PropagateEngine& {
        for (iter_t iter = 0; iter < num_iters; ++iter) {
            propagate<UdfUpdaterT>(udf_updater);
            merge_logs();

            if constexpr (profiling_mode) {
                ARTEA_INFO(fmt::format(
                    "Updater Iter {} ({}): Merged {} logs",
                    iter,
                    UdfUpdaterT::updater_name,
                    _merged_logs_count
                ));
            }
        }
        return *this;
    }

    /** @brief Accessor for the internal DelegateTable. */
    __attribute__((always_inline))
    auto get_log_table() -> log_table_t& {
        return _log_table;
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
      *   - routing_updater_t: Requires topk, candidate_queue_size (refining_graph is auto-provided from internal state)
      */
    template <typename UpdaterT, typename... Args>
    auto make_updater(Args&&... args) -> UpdaterT {
        using triangle_updater_t = typename RefinerTraitsT::triangle_updater_t;
        using pruning_updater_t  = typename RefinerTraitsT::pruning_updater_t;
        using reverse_updater_t  = typename RefinerTraitsT::reverse_updater_t;
        using random_updater_t   = typename RefinerTraitsT::random_updater_t;
        using routing_updater_t  = typename RefinerTraitsT::routing_updater_t;
        using truncate_updater_t = typename RefinerTraitsT::truncate_updater_t;

        const auto& vecs_arr = _refining_graph->get_vecs_data();
        auto& log_table = _log_table;
        const auto max_nbr_size = _refining_graph->layer_config().max_nbr_size();
        const auto num_vertices = _refining_graph->get_num_vertices();

        if constexpr (std::is_same_v<UpdaterT, triangle_updater_t>) {
            return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph, std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<UpdaterT, pruning_updater_t>) {
            return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph, std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<UpdaterT, reverse_updater_t>) {
            return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph);
        } else if constexpr (std::is_same_v<UpdaterT, random_updater_t>) {
            return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph, num_vertices, std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<UpdaterT, routing_updater_t>) {
            return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph, std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<UpdaterT, truncate_updater_t>) {
            return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph, std::forward<Args>(args)...);
        } else {
            ARTEA_ERROR("Unsupported updater type");
        }
    }

private:

    /** @brief Operation log table for recording graph operations during propagation. */
    log_table_t _log_table;

    /** @brief Pointer to the descent graph being operated on. */
    refining_graph_t* _refining_graph;

    /** @brief Distance function reference. */
    const dist_func_t& _dist_func;

    /** @brief Total count of merged logs (accumulated across all merge_logs() calls). */
    size_t _merged_logs_count = 0;

};  // class PropagateEngine

}   // namespace cpu
}   // namespace artea
