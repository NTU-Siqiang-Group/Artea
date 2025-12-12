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
 * @FilePath: /Artea/include/artea/cpu/utils/direction.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace artea {
namespace cpu {

/** @brief Direction type for graph edges. */
enum class graph_direction_t : uint16_t {
    IN = 0,
    OUT = 1,
    HIBRID = 2
};

/** @brief Reverse the direction.
  * @param dir The original direction.
  * @return The reversed direction.
  */
__attribute__((always_inline))
auto reverse(const graph_direction_t dir) -> graph_direction_t {
    return (dir == graph_direction_t::IN) ? graph_direction_t::OUT : graph_direction_t::IN;
}

/** @brief Operation direction type for graph operations. */
enum class op_direction_t : uint8_t {
    IN = 0,
    OUT = 1
};

/** @brief Reverse the direction.
  * @param dir The original direction.
  * @return The reversed direction.
  */
__attribute__((always_inline))
auto reverse(const op_direction_t dir) -> op_direction_t {
    return (dir == op_direction_t::IN) ? op_direction_t::OUT : op_direction_t::IN;
}

/** @brief Convert graph_direction_t to op_direction_t.
  * @param dir The graph direction.
  * @return The corresponding operation direction.
  * @throws std::invalid_argument if the input direction is HIBRID.
  */
__attribute__((always_inline))
auto convert(const graph_direction_t dir) -> op_direction_t {
    if (dir == graph_direction_t::IN) {
        return op_direction_t::IN;
    } else if (dir == graph_direction_t::OUT) {
        return op_direction_t::OUT;
    } else {
        throw std::invalid_argument("Cannot convert HIBRID graph_direction_t to op_direction_t.");
    }
}

/** @brief Convert op_direction_t to graph_direction_t.
  * @param dir The operation direction.
  * @return The corresponding graph direction.
  */
__attribute__((always_inline))
auto convert(const op_direction_t dir) -> graph_direction_t {
    if (dir == op_direction_t::IN) {
        return graph_direction_t::IN;
    } else {
        return graph_direction_t::OUT;
    }
}

}   // namespace cpu
}   // namespace artea