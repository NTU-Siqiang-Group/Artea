/*
 * @FilePath: /Artea/include/artea/cpu/index/graph_constructor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-22 14:50:08
 * @Date: 2025-11-15 20:36:29
 * @Description:
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

namespace artea {
namespace cpu {

template <typename ConstructorTraitsT>
class GraphConstructor {

    using vertex_num_t = typename ConstructorTraitsT::vertex_num_t;
    using vertex_id_t = typename ConstructorTraitsT::vertex_id_t;
    using vec_ele_t = typename ConstructorTraitsT::vec_ele_t;
    using iter_t = typename ConstructorTraitsT::iter_t;
    using index_graph_t = typename ConstructorTraitsT::index_graph_t;
    using vector_dataset_t = typename ConstructorTraitsT::vector_dataset_t;
    using dist_func_t = typename ConstructorTraitsT::dist_func_t;
    using propagate_engine_t = typename ConstructorTraitsT::propagate_engine_t;
    using rng_updater_t = typename ConstructorTraitsT::rng_updater_t;

public:
    GraphConstructor(vector_dataset_t& dataset) : _dataset(dataset) {}

    /** @brief construct a new graph */
    auto construct_graph(
        const vertex_num_t reserved_nbrs_size,
        const iter_t num_iters
    ) -> index_graph_t {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(_dataset.get_num_base_vecs());
        index_graph_t index_graph(
            /* num_vertices =       */ num_vertices,
            /* reserved_nbrs_size = */ reserved_nbrs_size
        );

        propagate_engine_t propagate_engine(index_graph);
        dist_func_t dist_func(_dataset.get_vec_dim());
        rng_updater_t rng_updater(
            /* dist_func = */ dist_func,
            /* vecs_arr =  */ _dataset.get_base_vecs(),
            /* log_table = */ propagate_engine.get_log_table()
        );

        propagate_engine.template run<rng_updater_t, true>(
            /* num_iters =   */ num_iters,
            /* udf_updater = */ rng_updater
        );
        return index_graph;
    }

private:

    /** @brief Vector dataset */
    vector_dataset_t& _dataset;

};  // class GraphConstructor


}   // namespace cpu
}   // namespace artea