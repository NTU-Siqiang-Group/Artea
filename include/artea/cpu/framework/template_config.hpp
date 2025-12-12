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
 * @FilePath: /Artea/include/artea/cpu/framework/template_config.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>
#include <mutex>

#include <tbb/spin_mutex.h>

#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/tbb_buffer.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

enum class buffer_policy_t : uint8_t {
    LOCKED_BUFFER_WITH_MUTEX = 0,
    LOCKED_BUFFER_WITH_SPINLOCK = 1,
    TBB_CONCURRENT_BUFFER = 2,
    THREAD_LOCAL_BUFFER = 3
};

template <
    buffer_policy_t BufferPolicy,
    typename T,
    std::size_t buf_capacity,
    typename lock_t
>
struct BufferSelector;

template <typename T, std::size_t buf_capacity>
struct BufferSelector<buffer_policy_t::LOCKED_BUFFER_WITH_MUTEX, T, buf_capacity, std::mutex> {
    using type = LockedBuffer<T, buf_capacity, std::mutex>;
};

template <typename T, std::size_t buf_capacity>
struct BufferSelector<buffer_policy_t::LOCKED_BUFFER_WITH_SPINLOCK, T, buf_capacity, tbb::spin_mutex> {
    using type = LockedBuffer<T, buf_capacity, tbb::spin_mutex>;
};

template <typename T, std::size_t buf_capacity>
struct BufferSelector<buffer_policy_t::TBB_CONCURRENT_BUFFER, T, buf_capacity, void> {
    using type = TbbBuffer<T, buf_capacity>;
};

template <typename vertex_num_t, typename vec_ele_t>
struct TemplateContext {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

};  //

}   // namespace cpu
}   // namespace artea