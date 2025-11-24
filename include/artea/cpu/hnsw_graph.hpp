/*
 * @FilePath: /Artea/include/artea/cpu/hnsw_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-23 10:02:40
 * @Date: 2025-11-20 18:52:43
 * @Description:
 */

#pragma once

#include <vector>

namespace artea {
namespace cpu {

/**
 * @brief The mini index graph class.
 * @tparam{vertex_num_t} The type of vertex number.
 * @tparam{vec_ele_t} The type of vector element.
 * @tparam{direction} The direction of the graph (IN, OUT, HIBRID).
 */
template <
    typename vertex_num_t,
    typename vec_ele_t,
    direction_t direction = direction_t::OUT
>
class MiniIndexGraph {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

public:
    MiniIndexGraph(
        const vertex_num_t num_vertices,
        const vertex_num_t edges_limit
    ) :
        _num_vertices(num_vertices),
        _edges_limit(edges_limit)
    {

    }

private:

    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Number of neighbors per vertex. */
    vertex_num_t _edges_limit;

    /** @brief */

};

}   // namespace cpu
}   // namespace artea
