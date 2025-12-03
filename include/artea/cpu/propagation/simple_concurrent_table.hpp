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
 * @FilePath: /Artea/include/artea/cpu/propagation/simple_dlt.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <utility>
#include <stdexcept>

#include <artea/cpu/containers/spinlock_buffer.hpp>
#include <artea/cpu/propagation/concurrent_table.hpp>
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
class SimpleConcurrentTable final :
    public ConcurrentTable<
        vertex_num_t,
        vec_ele_t,
        SimpleDelegateEntry<vertex_num_t, vec_ele_t>,
        SimpleConcurrentTable<vertex_num_t, vec_ele_t, lock_t, buf_size>
    >
{
    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
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

public:

    /** @brief Construct a new Simple DLT object. */
    SimpleConcurrentTable(vertex_num_t num_vertices) :
        base_class_t(num_vertices),
        _active_buffers(num_vertices),
        _shadow_buffers(num_vertices)
    {}

    /** @brief Append a delegate entry */
    auto append_entry_impl(
        const vertex_id_t src,
        const vertex_id_t dest,
        const distance_t dist
    ) -> bool override
    {

    }

    /** @brief Get the delegate entries of a vertex. */
    auto flush_entries_impl(
        const vertex_id_t vid
    ) -> de_rolist_t override {

    }

private:

    /** @brief buffer for (simple) delegate entries */
    std::vector<de_buf_t> _active_buffers;
    std::vector<de_buf_t> _shadow_buffers;

};  // class SimpleConcurrentTable

}   // namespace cpu
}   // namespace artea