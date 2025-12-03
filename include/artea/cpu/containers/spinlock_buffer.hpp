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
 * @FilePath: /Artea/include/artea/cpu/containers/spinlock_buffer.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>

namespace artea {
namespace cpu {

template <
    typename container_t,
    typename lock_t,
    std::size_t buf_size = 64
>
struct alignas(CACHE_LINE_SIZE) SpinlockBuffer {

    /** @brief Container protected by a spinlock */
    container_t container;

    /** @brief Spinlock for synchronizing access to the container */
    lock_t lock;

    /** @brief Default constructor that reserves buffer size */
    SpinlockBuffer() {
        container.reserve(buf_size);
    }

};  // struct SpinlockBuffer

}   // namespace cpu
}   // namespace artea