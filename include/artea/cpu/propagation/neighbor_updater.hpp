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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, typename BufferTraitsT, typename DerivedClassT>
class NeighborUpdater {

    using vertex_id_t = typename ComputerTraitsT::vertex_id_t;
    using vertex_num_t = typename ComputerTraitsT::vertex_num_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using vector_array_t = typename BufferTraitsT::vector_array_t;
    using nbr_t = typename BufferTraitsT::nbr_t;
    using nbr_arr_t = typename BufferTraitsT::nbr_arr_t;
    using log_table_t = typename BufferTraitsT::log_table_t;
    using dist_func_t = typename BufferTraitsT::dist_func_t;

public:

    NeighborUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_arr,
        log_table_t& log_table
    ) : _dist_func(dist_func), _vecs_arr(vecs_arr), _log_table(log_table) {}

protected:

    virtual auto _internal_check() -> std::tuple<bool, vertex_id_t, distance_t> = 0;

    /** @brief Distance function used for RNG checking. */
    const dist_func_t& _dist_func;

    /** @brief Reference to the vector array. */
    const vector_array_t& _vecs_arr;

    /** @brief Reference to the operation log table. */
    log_table_t& _log_table;

};  //  class NeighborUpdater

}   // namespace cpu
}   // namespace artea