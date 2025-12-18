/*
 * @FilePath: /Artea/include/artea/cpu/index/graph_constructor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-12-13 10:19:07
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

#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/propagation/propagate_engine.hpp>
#include <artea/cpu/propagation/rng_updater.hpp>
#include <artea/common/definitions.hpp>

namespace artea {
namespace cpu {

template <typename type_context_t>
class GraphConstructor {

    using distance_t = typename type_context_t::vec_ele_t;
    using vertex_id_t = typename type_context_t::vertex_num_t;
    using nbr_t = typename type_context_t::nbr_t;
    using nbr_arr_t = typename type_context_t::nbr_arr_t;
    using log_buffer_t = typename type_context_t::log_buffer_t;
    using log_table_t = typename type_context_t::log_table_t;
    using dist_func_t = typename type_context_t::dist_func_t;
    using rng_updater_t = typename type_context_t::rng_updater_t;
    using propagate_engine_t = typename type_context_t::propagate_engine_t;
    using index_graph_t = typename type_context_t::index_graph_t;
    using vector_dataset_t = typename type_context_t::vector_dataset_t;

public:
    GraphConstructor(vector_dataset_t& dataset) :
        _dataset(dataset) {}

    /** @brief construct a new graph */
    auto construct_graph(
        const vertex_num_t init_nbrs_size,
        const iter_t num_iters
    ) -> index_graph_t {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(_dataset.get_num_base_vecs());
        index_graph_t index_graph(
            /* num_vertices = */ num_vertices,
            /* reserved_nbrs_size = */ init_nbrs_size * 2
        );

        propagate_engine_t propagate_engine(index_graph);
        dist_func_t dist_func(_dataset.get_vec_dim());
        rng_updater_t rng_updater(
            /* dist_func = */ dist_func,
            /* vecs_arr = */ _dataset.get_base_vecs(),
            /* op_log_table = */ propagate_engine.get_log_table()
        );

        propagate_engine.template run<rng_updater_t, true>(
            /* num_iters = */ num_iters,
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