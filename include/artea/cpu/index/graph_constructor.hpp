/*
 * @FilePath: /Artea/include/artea/cpu/index/graph_constructor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-12-12 10:40:34
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
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN
>
class GraphConstructor {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;
    using log_buffer_t = LockedBuffer<Neighbor<vertex_id_t, distance_t>, 64, tbb::spin_mutex>;
    using log_table_t = NbrLogTable<vertex_num_t, vec_ele_t, log_buffer_t>;
    using dist_func_t = SIMDDistance<vec_ele_t, dist_type>;
    using rng_updater_t = RNGUpdater<vertex_num_t, vec_ele_t, log_table_t, dist_func_t>;
    using propagate_engine_t = PropagateEngine<vertex_num_t, vec_ele_t, log_buffer_t>;
    using index_graph_t = IndexGraph<vertex_num_t, vec_ele_t, graph_direction_t::HIBRID>;

public:
    GraphConstructor(VectorDataset<vertex_num_t, vec_ele_t>& dataset) :
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
    VectorDataset<vertex_num_t, vec_ele_t>& _dataset;

};  // class GraphConstructor


}   // namespace cpu
}   // namespace artea