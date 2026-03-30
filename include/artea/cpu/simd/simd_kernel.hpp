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
 * @FilePath: /Artea/include/artea/cpu/simd/simd_kernel.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <cassert>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename BaseTraitsT>
class SIMDKernel {

    static constexpr std::size_t SIMD_REGISTER_BITS = 512;
    static constexpr std::size_t SIMD_REGISTER_BYTES = SIMD_REGISTER_BITS / 8;  // 64
    // Number of elements that can be processed in a single SIMD register
    // e.g. we can process 16 elements per chunk for float32 type
    static constexpr std::size_t SIMD_CHUNK_SIZE = [] {
        static_assert(std::is_same_v<vec_ele_t, float>, "SIMDDistance currently supports float type only");
        static_assert(
            SIMD_REGISTER_BYTES % sizeof(vec_ele_t) == 0,
            "SIMD register size must be a multiple of the element size for this utility."
        );
        return SIMD_REGISTER_BYTES / sizeof(vec_ele_t);
    }();    // 16 for float32 type

public:


};  // class SIMDKernel

}   // namespace cpu
}   // namespace artea