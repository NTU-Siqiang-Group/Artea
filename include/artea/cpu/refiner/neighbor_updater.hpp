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
 * @FilePath: /Artea/include/artea/cpu/refiner/neighbor_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Neighbor updater base class for edge generation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace artea {
namespace cpu {

template <typename RefinerTraitsT, typename RefiningGraphT, typename DerivedClassT>
class NeighborUpdater {

    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using nbr_arr_checker_t = typename RefinerTraitsT::nbr_arr_checker_t;

public:

    NeighborUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const RefiningGraphT& refining_graph
    ) : _dist_func(dist_func), _vecs_data(vecs_data), _log_table(log_table), _refining_graph(refining_graph) {}

    /**
     * @brief Operator that delegates to the derived class's update_impl.
     *
     * This uses CRTP (Curiously Recurring Template Pattern) to call the derived
     * class's update_impl method without virtual function overhead.
     */
    __attribute__((always_inline))
    auto operator()(
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        static_cast<DerivedClassT*>(this)->update_impl(pivot_vid, origin_nbrs);

        #ifndef NDEBUG
        if (!nbr_arr_checker_t::full_check(origin_nbrs)) {
            ARTEA_ERROR(fmt::format(
                "Updater {} produced an invalid neighbor array for vertex {}.",
                DerivedClassT::updater_name,
                pivot_vid
            ));
        }
        #endif
    }

protected:

    /** @brief Distance function used for RNG checking. */
    const dist_func_t& _dist_func;

    /** @brief Reference to the vector array. */
    const vector_array_t& _vecs_data;

    /** @brief Reference to the operation log table. */
    log_table_t& _log_table;

    /** @brief Reference to the descent graph for neighbor overflow check. */
    const RefiningGraphT& _refining_graph;

};  //  class NeighborUpdater

}   // namespace cpu
}   // namespace artea