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
 * @FilePath: /Artea/include/artea/cpu/propagation/delegate_entry_pool.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>
#include <functional>

#include <tbb/tbb.h>
#include <tbb/enumerable_thread_specific.h>

#include <artea/cpu/propagation/delegate_entry.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename delegate_entry_t
>
struct ThreadLocalDEP {

    using pool_size_t = vertex_num_t;
    using entry_idx_t = pool_size_t;

    /** @brief The pool of delegate entries for the thread. */
    mmap_container_t<delegate_entry_t> storage;

    /** @brief The current index in the pool. */
    entry_idx_t cur_idx = 0;

    /** @brief Reset the pool index to zero. */
    __attribute__((always_inline))
    auto reset() -> void {

    }

    /** @brief Get the next available delegate entry from the pool. */
    __attribute__((always_inline))
    auto get_next_entry() -> delegate_entry_t* {


    }
};

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename delegate_entry_t
>
class DelegateEntryPool {

public:

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using pool_size_t = vertex_num_t;

    explicit DelegateEntryPool(const pool_size_t thd_pool_size) : _thd_pool_size(thd_pool_size) {}

    /** @brief Get the pool size for each thread. */
    __attribute__((always_inline))
    auto get_thd_pool_size() const -> pool_size_t {
        return _thd_pool_size;
    }

private:

    /** @brief number of entries in the pool of single thread */
    pool_size_t _thd_pool_size;

}; // class DelegateEntryPool

}   // namespace cpu
}   // namespace artea