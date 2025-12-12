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
 * @FilePath: /Artea/include/artea/cpu/utils/clear_cache.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>

#include <artea/definitions.hpp>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

auto clear_cpu_cache() -> void {
    // artea::logger.info("Clearing CPU cache by allocating large dummy buffer...");
    constexpr std::size_t CACHE_FLUSH_BYTES = 1024 * 1024 * 1024;
    constexpr std::size_t NUM_INTS = CACHE_FLUSH_BYTES / sizeof(int);

    std::vector<int> dummy(NUM_INTS);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, NUM_INTS),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const_cast<volatile int&>(dummy[i]) = static_cast<int>(i);
            }
        }
    );

    std::atomic_thread_fence(std::memory_order_seq_cst);

    artea::logger.success("CPU cache cleared.");
}

}   // namespace cpu
}   // namespace artea