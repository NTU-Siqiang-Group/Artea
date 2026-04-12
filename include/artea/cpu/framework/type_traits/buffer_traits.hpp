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

namespace artea {
namespace cpu {

/* ------ Forward Declarations ------ */

template <typename T, std::size_t BufCapacity, typename LockT> struct LockedBuffer;
template <typename T, std::size_t BufCapacity> struct TbbBuffer;
template <typename BufferTraitsT> class NbrLogTable;

/* ------ Buffer Definition ------ */

enum class BufferPolicyT : uint8_t {
    LOCKED_BUFFER_WITH_MUTEX = 0,
    LOCKED_BUFFER_WITH_SPINLOCK = 1,
    TBB_CONCURRENT_BUFFER = 2
};  // enum class BufferPolicyT

template <
    BufferPolicyT BufferPolicy,
    typename T,
    std::size_t BufCapacity
>
struct BufferSelector;

template <typename T, std::size_t BufCapacity>
struct BufferSelector<BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX, T, BufCapacity> {
    using type = LockedBuffer<T, BufCapacity, std::mutex>;
};

template <typename T, std::size_t BufCapacity>
struct BufferSelector<BufferPolicyT::LOCKED_BUFFER_WITH_SPINLOCK, T, BufCapacity> {
    using type = LockedBuffer<T, BufCapacity, tbb::spin_mutex>;
};

template <typename T, std::size_t BufCapacity>
struct BufferSelector<BufferPolicyT::TBB_CONCURRENT_BUFFER, T, BufCapacity> {
    using type = TbbBuffer<T, BufCapacity>;
};

template <
    typename BaseTraitsT,
    BufferPolicyT BufferPolicy,
    std::size_t BufCapacity
>
struct BufferTraits : virtual public BaseTraitsT {

private:

    /** ------ Self Traits ------ **/
    using buffer_traits_t = BufferTraits<BaseTraitsT, BufferPolicy, BufCapacity>;

public:

    using base_traits_t = BaseTraitsT;

    /** @brief Type for buffer elements. */
    using bnbr_t = typename BaseTraitsT::bnbr_t;

    using buffer_policy_t = BufferPolicyT;

    /** ------ Selected Buffer Type ------ **/

    /** @brief Type for neighbor log buffers. */
    using log_buffer_t = typename BufferSelector<BufferPolicy, bnbr_t, BufCapacity>::type;

     /** @brief Type for the container used within the neighbor buffer. */
    using log_container_t = typename log_buffer_t::container_t;

    /** @brief Type for neighbor log tables. */
    using log_table_t = NbrLogTable<buffer_traits_t>;

    /** @brief Buffer policy used for the buffers. */
    static constexpr BufferPolicyT buffer_policy = BufferPolicy;

    /** @brief Capacity of each buffer. */
    static constexpr std::size_t buf_capacity = BufCapacity;

};  // struct BufferTraits

}   // namespace cpu
}   // namespace artea