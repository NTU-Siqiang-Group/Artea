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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/truncate_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Truncate updater: trims each vertex's neighbor array to max_nbr_size.
 */

#pragma once

#include <cstddef>

namespace artea {
namespace cpu {

template <typename EdgeGeneratorTraitsT>
class TruncateUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<TruncateUpdater<EdgeGeneratorTraitsT>> {

    using vertex_id_t    = typename EdgeGeneratorTraitsT::vertex_id_t;
    using vertex_num_t   = typename EdgeGeneratorTraitsT::vertex_num_t;
    using nbr_arr_t      = typename EdgeGeneratorTraitsT::nbr_arr_t;
    using log_table_t    = typename EdgeGeneratorTraitsT::log_table_t;
    using dist_func_t    = typename EdgeGeneratorTraitsT::dist_func_t;
    using vector_array_t = typename EdgeGeneratorTraitsT::vector_array_t;
    using conv_graph_index_t   = typename EdgeGeneratorTraitsT::conv_graph_index_t;
    using base_class_t   = typename EdgeGeneratorTraitsT::template neighbor_updater_t<TruncateUpdater<EdgeGeneratorTraitsT>>;

public:
    static constexpr const char* updater_name = "truncate_updater";

    TruncateUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const conv_graph_index_t& flat_graph
    ) : base_class_t(dist_func, vecs_data, log_table, flat_graph) {}

    /**
     * @brief Truncate the neighbor array of a pivot vertex to at most max_nbr_size entries.
     *
     * Since flat_graph neighbor arrays are maintained in sorted order (closest first),
     * truncation simply drops the tail entries beyond max_nbr_size, retaining only
     * the closest neighbors.
     *
     * @param pivot_vid The vertex ID whose neighbor array is being truncated (unused).
     * @param origin_nbrs The neighbor array to truncate in-place. If its size is already
     *                    <= max_nbr_size, it is left unchanged.
     *
     * @note This updater does not compute distances, write logs, or modify neighbor
     *       distances/flags. It is intended as a post-processing pass after graph
     *       construction to enforce the max_nbr_size capacity constraint.
     */
    __attribute__((always_inline))
    auto update_impl(
        const vertex_id_t /* pivot_vid */,
        nbr_arr_t& origin_nbrs
    ) -> void {
        const vertex_num_t max_sz = this->_flat_graph.layer_config().max_nbr_size();
        if (origin_nbrs.size() > max_sz) {
            origin_nbrs.resize(max_sz);
        }
    }

};  // class TruncateUpdater

}   // namespace cpu
}   // namespace artea
