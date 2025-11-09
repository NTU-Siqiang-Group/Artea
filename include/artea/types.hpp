/*
 * @FilePath: /Artea/include/artea/types.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-09 17:28:59
 * @Date: 2025-10-18 16:36:47
 * @Description: 
 */

#pragma once

#include <cstdint>

namespace artea {

using vec_dim_t = uint32_t;

using part_num_t = uint16_t;

using part_id_t = part_num_t;

using iter_t = uint32_t;

/** @brief Neighbor structure for storing vertex ID and distance. */
template <typename vertex_num_t, typename vec_ele_t>
struct alignas(8) Neighbor {   // 8 bytes

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;

    vertex_id_t dest;
    distance_t distance;
    // bool new_flag;
    // uint8_t padding[7];
    
};  // struct Neighbor

/** @brief Comparator for Neighbor, lambda function. */
template <typename vertex_num_t, typename vec_ele_t>
constexpr auto NeighborComparator = [](
    const Neighbor<vertex_num_t, vec_ele_t>& a, 
    const Neighbor<vertex_num_t, vec_ele_t>& b
) -> bool {
    return a.distance < b.distance or
          (a.distance == b.distance && a.dest < b.dest);
};  // constexpr auto NeighborComparator

}   // namespace artea
