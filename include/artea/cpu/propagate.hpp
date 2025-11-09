/*
 * @FilePath: /Artea/include/artea/cpu/propagate.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-10 01:15:38
 * @Date: 2025-11-09 20:46:35
 * @Description: 
 */

#pragma once

#include <artea/cpu/index_graph.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/recommended_nn.hpp>
#include <artea/cpu/random_nn.hpp>
#include <artea/cpu/types.hpp>

template <typename vertex_num_t, typename vec_ele_t>
class Propagate {

public:
    using distance_t = IndexGraph<vertex_num_t, vec_ele_t>::distance_t;
    using vertex_id_t = IndexGraph<vertex_num_t, vec_ele_t>::vertex_id_t;
    using nbr_t = IndexGraph<vertex_num_t, vec_ele_t>::nbr_t;
    using nbr_arr_t = IndexGraph<vertex_num_t, vec_ele_t>::nbr_arr_t;

public:
    Propagate(
        VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        std::vector<nbr_arr_t>& nbrs_arr,
        vec_dim_t vec_dim
    ) : _vecs_arr(vecs_arr), _nbrs_arr(nbrs_arr), _vec_dim(vec_dim)
    {
    }

    virtual auto init_strategy() -> void = 0;

    virtual auto update_strategy() -> void = 0;

    virtual auto conflict_check() -> bool = 0;

protected:

    /** @brief The dimension of each vector. */
    vec_dim_t _vec_dim;

    /** @brief The neighbors array of each vertex. */
    std::vector<nbr_arr_t>& _nbrs_arr;

    /** @brief vector array to be processed */
    VectorArray<vertex_num_t, vec_ele_t>& _vecs_arr;

};  // class Propagate
