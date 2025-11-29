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
 * @FilePath: /Artea/include/artea/cpu/utils/parallel.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <tbb/task_arena.h>

namespace artea {
namespace cpu {

/** @brief Get the maximum number of threads used by TBB.
  * @return The number of threads used by TBB.
  */
__attribute__((always_inline))
auto tbb_max_num_threads() -> int {
    return tbb::task_arena::max_concurrency();
}

}   // namespace cpu
}   // namespace artea