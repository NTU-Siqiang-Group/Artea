/*
 * @FilePath: /Artea/include/artea/definitions.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-19 18:40:47
 * @Date: 2025-10-18 16:36:47
 * @Description:
 */

#pragma once

#include <cstdint>
#include <vector>

#include <artea/utils.hpp>

namespace artea {

using vec_dim_t = uint32_t;

using part_num_t = uint16_t;

using part_id_t = part_num_t;

using iter_t = uint32_t;

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

constexpr std::size_t AVX512_ALIGNMENT = 64;

/** @brief a container with AVX-512 alignment */
template <typename T>
using avx512_container_t = std::vector<T, AlignedAllocator<T, AVX512_ALIGNMENT>>;

}   // namespace artea