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
 * @FilePath: /Artea/include/artea/cpu/utils/nbr_arr_checker.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

#include <artea/definitions.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t>
class NbrArrChecker {

    using nbr_t = Neighbor<uint32_t, float>;
    using nbr_arr_t = cache_aligned_container_t<Neighbor<vertex_num_t, vec_ele_t>>;

public:

    static auto no_nan_check(const nbr_arr_t& nbrs) -> bool {
        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            if (is_nan_distance(nbrs[i].get_distance())) {
                return false;
            }
        }
        return true;
    }

    static auto no_removed_check(const nbr_arr_t& nbrs) -> bool {
        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            if (nbrs[i].is_removed()) {
                return false;
            }
        }
        return true;
    }

    static auto no_duplicate_check(const nbr_arr_t& nbrs) -> bool {
        std::vector<vertex_num_t> seen_ids;
        seen_ids.reserve(nbrs.size());

        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            vertex_num_t nbr_id = nbrs[i].get_id();
            if (std::find(seen_ids.begin(), seen_ids.end(), nbr_id) != seen_ids.end()) {
                return false;
            }
            seen_ids.push_back(nbr_id);
        }
        return true;
    }

    static auto distance_order_check(const nbr_arr_t& nbrs) -> bool {
        for (std::size_t i = 1; i < nbrs.size(); ++i) {
            if (NeighborDistanceComparator<vertex_num_t, vec_ele_t>(nbrs[i], nbrs[i - 1])) {
                return false;
            }
        }
        return true;
    }

};  // class NbrArrChecker

}   // namespace cpu
}   // namespace artea