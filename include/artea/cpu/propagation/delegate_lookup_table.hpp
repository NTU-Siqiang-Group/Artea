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
 * @FilePath: /Artea/include/artea/cpu/propagation/delegate_lookup_table.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <mutex>
#include <memory>
#include <unordered_map>

#include <artea/cpu/propagation/delegate_entry.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t>
class DelegateLookupTable {

public:

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using delegate_entry_t = DelegateEntry<vertex_num_t, vec_ele_t>;

private:

    /** @brief number of vertices */
    vertex_num_t _num_vertices;



};  // class DelegateLookupTable

}   // namespace cpu
}   // namespace artea