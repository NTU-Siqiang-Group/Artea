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

#include <algorithm>
#include <cstddef>
#include <vector>
#include <stdexcept>

namespace artea {
namespace cpu {

template <typename BaseTraitsT>
class NbrArrChecker {

    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vertex_id_t = typename BaseTraitsT::vertex_id_t;
    using vec_ele_t = typename BaseTraitsT::vec_ele_t;
    using distance_t = typename BaseTraitsT::distance_t;
    using nbr_t = typename BaseTraitsT::nbr_t;
    using nbr_arr_t = typename BaseTraitsT::nbr_arr_t;
    using nbr_dist_comp_t = typename BaseTraitsT::nbr_dist_comp_t;

    constexpr static nbr_dist_comp_t nbr_dist_comp {};

public:

    static auto invalid_id_suffix_check(const nbr_arr_t& nbrs) -> bool {
        bool seen_invalid_id = false;
        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            const vertex_id_t nbr_id = nbrs[i].get_id();
            const bool is_invalid_id = nbr_id == BaseTraitsT::invalid_vertex_id;
            if (is_invalid_id) {
                seen_invalid_id = true;
            } else if (seen_invalid_id) {
                ARTEA_ERROR("Neighbor array contains a valid neighbor after invalid_vertex_id suffix begins.");
                return false;
            }
        }
        return true;
    }

    static auto invalid_id_suffix_check(const vertex_id_t* nbrs, const vertex_num_t nbr_count) -> bool {
        bool seen_invalid_id = false;
        for (vertex_num_t i = 0; i < nbr_count; ++i) {
            const bool is_invalid_id = nbrs[i] == BaseTraitsT::invalid_vertex_id;
            if (is_invalid_id) {
                seen_invalid_id = true;
            } else if (seen_invalid_id) {
                ARTEA_ERROR("Neighbor row contains a valid neighbor after invalid_vertex_id suffix begins.");
                return false;
            }
        }
        return true;
    }

    static auto no_nan_check(const nbr_arr_t& nbrs) -> bool {
        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            if (BaseTraitsT::is_nan_distance(nbrs[i].get_distance())) {
                ARTEA_ERROR("Neighbor array contains NaN distances before applying logs.");
                // throw std::runtime_error("Error: Neighbor array contains NaN distances before applying logs.");
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
            if (nbr_id == BaseTraitsT::invalid_vertex_id) {
                break;
            }
            if (std::find(seen_ids.begin(), seen_ids.end(), nbr_id) != seen_ids.end()) {
                ARTEA_ERROR("Neighbor array contains duplicate neighbors before applying logs.");
                return false;
            }
            seen_ids.push_back(nbr_id);
        }
        return true;
    }

    static auto distance_order_check(const nbr_arr_t& nbrs) -> bool {
        for (std::size_t i = 1; i < nbrs.size(); ++i) {
            if (nbrs[i - 1].get_id() == BaseTraitsT::invalid_vertex_id ||
                nbrs[i].get_id() == BaseTraitsT::invalid_vertex_id) {
                break;
            }
            if (nbr_dist_comp(nbrs[i], nbrs[i - 1])) {
                ARTEA_ERROR("Neighbor array is not sorted by distance before applying logs.");
                return false;
            }
        }
        return true;
    }

    static auto full_check(const nbr_arr_t& nbrs) -> bool {
        return invalid_id_suffix_check(nbrs) &&
               no_nan_check(nbrs) &&
               no_duplicate_check(nbrs) &&
               distance_order_check(nbrs);
    }

};  // class NbrArrChecker

}   // namespace cpu
}   // namespace artea