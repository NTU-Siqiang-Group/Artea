/*
 * @FilePath: /Artea/include/artea/cpu/constructor/graph_constructor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-02-10 21:54:07
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
    using ratio_t = typename ConstructorTraitsT::ratio_t;
    using flat_graph_t = typename ConstructorTraitsT::flat_graph_t;
    using vector_dataset_t = typename ConstructorTraitsT::vector_dataset_t;
    using dist_func_t = typename ConstructorTraitsT::dist_func_t;
    using propagate_engine_t = typename ConstructorTraitsT::propagate_engine_t;
    using triangle_updater_t = typename ConstructorTraitsT::triangle_updater_t;

public:
    GraphConstructor(vector_dataset_t& dataset) : _dataset(dataset) {}

    /** @brief construct a new graph */
    auto construct_graph(
        const vertex_num_t max_nbr_size,
        const vertex_num_t reserved_nbr_size,
        const ratio_t scale_coeffs = 1.0,
        const ratio_t shited_coeffs = 0.0
    ) -> flat_graph_t {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(_dataset.get_num_base_vecs());
        flat_graph_t flat_graph(
            /* vecs_data =          */ _dataset.get_base_vecs(),
            /* num_vertices =       */ num_vertices,
            /* max_nbr_size =       */ max_nbr_size,
            /* reserved_nbr_size =  */ reserved_nbr_size
        );

        propagate_engine_t propagate_engine(num_vertices);
        propagate_engine.set_graph(flat_graph);

        dist_func_t dist_func(_dataset.get_vec_dim());
        triangle_updater_t triangle_updater(
            /* dist_func =      */ dist_func,
            /* vecs_arr =       */ _dataset.get_base_vecs(),
            /* log_table =      */ propagate_engine.get_log_table(),
            /* max_nbr_size =   */ max_nbr_size,
            /* scale_coeffs =   */ scale_coeffs,
            /* shifted_coeffs = */ shited_coeffs
        );

        propagate_engine.run<triangle_updater_t>(
            /* num_iters =   */ num_iters,
            /* udf_updater = */ triangle_updater
        );
        return flat_graph;
    }

private:

    /** @brief Vector dataset */
    vector_dataset_t& _dataset;

};  // class GraphConstructor


}   // namespace cpu
}   // namespace artea
