/*
 * @FilePath: /Artea/include/artea/cpu/descent_engine.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:52:16
 * @Date: 2025-11-09 20:46:35
 * @Description:
 */

#pragma once

#include <utility>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/containers/vector_array.hpp>

#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>
#include <artea/common/element_pos.hpp>

template <
    typename vertex_num_t,
    typename vec_ele_t,
    DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN,
    typename derived_class
>
class DescentEngine {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

public:

    DescentEngine(IndexGraph<vertex_num_t, vec_ele_t>& graph) :
        _vecs_arr(graph.get_vecs_arr()),
        _nbrs_arr(graph.get_nbrs_arr()),
        _num_vertices(graph.get_num_vertices()),
        _vec_dim(graph.get_vecs_arr().get_vec_dim())
    {
    }

    /** @brief Initialize the neighbors of each vertex in the graph. */
    auto initialize() -> void {
        // intialize neighbors array with random_seq
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, this->_num_vertices),
        [this](const tbb::blocked_range<vertex_id_t>& r) {
            for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                static_cast<derived_class*>(this)->initialize_op(vid);
            }
        });
    }

    /** @brief Update the neighbors of each vertex in the graph with user-defined ``propagate_op`` logic.
      * @param cur_iter The current iteration number.
    */
    auto propagate(const iter_t cur_iter) -> void {
        // update neighbors array with recom_nbr
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, this->_num_vertices),
        [this](const tbb::blocked_range<vertex_id_t>& r) {
            for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                static_cast<derived_class*>(this)->propagate_op(vid, cur_iter);
            }
        });
    }

    auto run(const iter_t num_iters) -> void {
        initialize();
        for (iter_t cur_iter = 0; cur_iter < num_iters; ++cur_iter) {
            propagate(cur_iter);
        }
    }

protected:

    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Number of neighbors per vertex. */
    vertex_num_t _edges_per_vertex;

    /** @brief The dimension of each vector. */
    const vec_dim_t _vec_dim;

    /** @brief The neighbors array of each vertex.
      * @note The neighbors array of each vertex can be only modified by
      *       the thread that process the corresponding vertex, thus the
      *       neighbors array of each vertex is thread-race free.
      */
    std::vector<nbr_arr_t>& _nbrs_arr;

    /** @brief vector array to be processed */
    VectorArray<vertex_num_t, vec_ele_t>& _vecs_arr;

};  // class DescentEngine
