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
 * @FilePath: /Artea/include/artea/cpu/framework/buffer_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>
#include <mutex>
#include <tbb/spin_mutex.h>

#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/tbb_buffer.hpp>


namespace artea {
namespace cpu {

/* ------ Forward Declarations ------ */

template <typename T, std::size_t BufCapacity, typename LockT> struct LockedBuffer;
template <typename T, std::size_t BufCapacity> struct TbbBuffer;
template <typename BufferTraitsT> class NbrLogTable;

/* ------ Buffer Definition ------ */

enum class buffer_policy_t : uint8_t {
    LOCKED_BUFFER_WITH_MUTEX = 0,
    LOCKED_BUFFER_WITH_SPINLOCK = 1,
    TBB_CONCURRENT_BUFFER = 2
};  // enum class buffer_policy_t

template <
    buffer_policy_t BufferPolicy,
    typename T,
    std::size_t BufCapacity
>
struct BufferSelector;

template <typename T, std::size_t BufCapacity>
struct BufferSelector<buffer_policy_t::LOCKED_BUFFER_WITH_MUTEX, T, BufCapacity> {
    using type = LockedBuffer<T, BufCapacity, std::mutex>;
};

template <typename T, std::size_t BufCapacity>
struct BufferSelector<buffer_policy_t::LOCKED_BUFFER_WITH_SPINLOCK, T, BufCapacity> {
    using type = LockedBuffer<T, BufCapacity, tbb::spin_mutex>;
};

template <typename T, std::size_t BufCapacity>
struct BufferSelector<buffer_policy_t::TBB_CONCURRENT_BUFFER, T, BufCapacity> {
    using type = TbbBuffer<T, BufCapacity>;
};

template <
    typename BaseTraitsT,
    buffer_policy_t BufferPolicy,
    std::size_t BufCapacity
>
struct BufferTraits : virtual public BaseTraitsT {

private:

    /** ------ Self Traits ------ **/
    using buffer_traits_t = BufferTraits<BaseTraitsT, BufferPolicy, BufCapacity>;

public:
    /** @brief Type for buffer elements. */
    using nbr_t = BaseTraitsT::nbr_t;

    /** ------ Selected Buffer Type ------ **/

    /** @brief Type for neighbor buffers. */
    using log_buffer_t = typename BufferSelector<BufferPolicy, nbr_t, BufCapacity>::type;

     /** @brief Type for the container used within the neighbor buffer. */
    using log_container_t = typename log_buffer_t::container_t;

    /** @brief Type for neighbor log tables. */
    using log_table_t = NbrLogTable<buffer_traits_t>;

};  // struct BufferTraits

}   // namespace cpu
}   // namespace artea