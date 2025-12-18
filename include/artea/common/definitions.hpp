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
 * @FilePath: /Artea/include/artea/common/definitions.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>

namespace artea {

enum class device_t {
    CPU = 0,
    GPU = 1
};

template <typename vertex_num_t, typename vec_ele_t>
struct alignas(8) VertexProperty {   // 8 bytes

    using distance_t = vec_ele_t;

    vertex_num_t indegree;
    distance_t avg_distance;

};  // struct VertexProperty

template <typename T>
constexpr auto get_max_value() -> T {
    return std::numeric_limits<T>::max();
}

template <typename vertex_id_t>
constexpr auto invalid_vertex_id() -> vertex_id_t {
    return get_max_value<vertex_id_t>();
}

template <typename vec_ele_t>
constexpr auto nan_distance() -> vec_ele_t {
    return std::numeric_limits<vec_ele_t>::quiet_NaN();
}

template <typename distance_t>
__attribute__((always_inline))
constexpr auto is_nan_distance(const distance_t dist) -> bool {
    return std::isnan(dist);
}

}   // namespace artea