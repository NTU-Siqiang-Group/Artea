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
 * @FilePath: /Artea/include/artea/cpu/propagation/rng_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

#include <artea/cpu/propagation/graph_op_log.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/propagation/propagate_engine.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename log_table_t,
    typename dist_func_t
>
class RNGUpdater {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

public:
    RNGUpdater(
        const dist_func_t& dist_func,
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        log_table_t& log_table
    ) : _dist_func(dist_func), _vecs_arr(vecs_arr), _log_table(log_table) {}

    auto operator()(
        const vertex_id_t pivot_vid,
        const op_direction_t op_direction,
        nbr_arr_t& origin_nbrs,
        nbr_arr_t& retained_nbrs
    ) -> void {
        for (vertex_id_t i = 0; i < origin_nbrs.size(); ++i) {
            const nbr_t& ori_nbr = origin_nbrs[i];
            auto [passed, delegated_vid, new_edge_dist] = _rng_check(ori_nbr, retained_nbrs);
            if (passed) {
                retained_nbrs.push_back(ori_nbr);
            }
            else {
                vertex_id_t sacrificed_vid = ori_nbr.get_id();

                /** @brief add append operation
                  * append edge [delegated -- sacrificed] (bidirectional)
                  */
                _log_table.add_append_log(
                    /* executor_vid = */delegated_vid,
                    /* nbr_id = */sacrificed_vid,
                    /* new_edge_dist = */new_edge_dist,
                    /* direction = */op_direction
                );
                _log_table.add_append_log(
                    /* executor_vid = */sacrificed_vid,
                    /* nbr_id = */delegated_vid,
                    /* new_edge_dist = */new_edge_dist,
                    /* direction = */reverse(op_direction)
                );

                /** @brief add remove operation
                  * remove edge for sacrificed [delegated to pivot]
                  */
                _log_table.add_remove_log(
                    /* executor_vid = */sacrificed_vid,
                    /* nbr_id = */pivot_vid,
                    /* removed_edge_dist = */ori_nbr.get_distance(),
                    /* direction = */reverse(op_direction)
                );
            }
        }
    }

private:

    auto _rng_check(
        const nbr_t& ori_nbr,
        const nbr_arr_t& retained_nbrs
    ) -> std::tuple<bool, vertex_id_t, distance_t> {
        const vec_ele_t* ori_vec = _vecs_arr.get(ori_nbr.get_id());

        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            if (ori_nbr.is_old() and retained_nbrs[i].is_old()) {
                continue;
            }
            const nbr_t& retained_nbr = retained_nbrs[i];
            const vec_ele_t* retained_vec = _vecs_arr.get(retained_nbr.get_id());
            vec_ele_t dist_to_retained = _dist_func(ori_vec, retained_vec);

            if (dist_to_retained < ori_nbr.get_distance()) {
                // RNG conflict detected
                return std::make_tuple(false, retained_nbr.get_id(), dist_to_retained);
            }
        }
        return std::make_tuple(true, invalid_vertex_id<vertex_id_t>(), nan_distance<distance_t>());
    }

    /** @brief Distance function used for RNG checking. */
    const dist_func_t& _dist_func;

    /** @brief Reference to the vector array. */
    const VectorArray<vertex_num_t, vec_ele_t>& _vecs_arr;

    /** @brief Reference to the operation log table. */
    log_table_t& _log_table;

};  // class RNGUpdater

}   // namespace cpu
}   // namespace artea