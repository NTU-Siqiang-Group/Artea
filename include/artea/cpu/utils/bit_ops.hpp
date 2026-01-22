// Copyright 2026 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/utils/bit_ops.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#ifdef _MSC_VER
#include <intrin.h>
#endif

#pragma once

namespace artea {
namespace cpu {

// Helper function for bit scanning (Platform independent)
__attribute__((always_inline))
inline int count_trailing_zeros(uint64_t val) {
    #if defined(_MSC_VER)
        unsigned long index;
        _BitScanForward64(&index, val);
        return static_cast<int>(index);
    #else
        return __builtin_ctzll(val);
    #endif
}

}   //  namespace cpu
}   // namespace artea