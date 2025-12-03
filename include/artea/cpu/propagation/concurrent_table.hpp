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
 * @FilePath: /Artea/include/artea/cpu/propagation/delegate_lookup_table.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <mutex>
#include <memory>
#include <unordered_map>

#include <artea/cpu/propagation/delegate_entry.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename lock_t,
    std::size_t buf_size = 64
>
class ConcurrentTable {

public:

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    /** @brief delegate entry list type */
    using de_buf_t = SpinlockBuffer<
        std::vector<SimpleDelegateEntry<vertex_num_t, vec_ele_t>>,
        lock_t, buf_size
    >;
    using delegate_entry_t = SimpleDelegateEntry<vertex_num_t, vec_ele_t>;
    /** @brief base class type */
    using base_class_t = ConcurrentTable<
        vertex_num_t,
        vec_ele_t,
        delegate_entry_t,
        SimpleConcurrentTable<vertex_num_t, vec_ele_t, lock_t, buf_size>
    >;

    ConcurrentTable(vertex_num_t num_vertices) : _num_vertices(num_vertices) {}

    __attribute__((always_inline))
    auto append_entry(
        const vertex_id_t src,
        const vertex_id_t dest,
        const distance_t dist
    ) -> bool {

    }

    __attribute__((always_inline))
    auto flush_entries() -> de_rolist_t {

    }

private:

    /** @brief number of vertices */
    vertex_num_t _num_vertices;

    /** @brief buffer for delegate entries */
    std::vector<de_buf_t> _active_buffers;

    /** @brief shadow buffer for delegate entries during flushing */
    std::vector<de_buf_t> _shadow_buffers;

};  // class ConcurrentTable

}   // namespace cpu
}   // namespace artea